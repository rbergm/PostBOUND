from __future__ import annotations

import dataclasses
import multiprocessing as mp
import multiprocessing.connection
import os
import pathlib
import re
import signal
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from postbound.db import db, postgres
from postbound.db.systems import systems
from postbound.qal import qal
from postbound.experiments import workloads
from postbound.optimizer import data
from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import operators
from postbound.util import jsonize

ExhaustiveJoinOrderingLimit = 1000
QuerySlowdownToleranceFactor = 3
DefaultQueryTimeout = 120

OutputFileFormat = "join-order-runtimes-{label}.csv"
OutputFilePattern = r"join-order-runtimes-(?P<label>\w+).csv"
OutputDirectory = "results/query-runtime-variation/"

StopEvaluation = False
workloads.workloads_base_dir = "../workloads"


def skip_existing_results(workload: workloads.Workload) -> workloads.Workload:
    existing_results = set()
    result_directory = pathlib.Path(OutputDirectory)
    for result_file in result_directory.glob("join-order-runtimes-*.csv"):
        file_matcher = re.match(OutputFilePattern, result_file.name)
        if not file_matcher:
            continue
        label = file_matcher.group("label")
        if not label:
            continue
        existing_results.add(label)

    if not existing_results:
        return workload
    skipped_workload = workload.filter_by(lambda label, __: label not in existing_results)
    print(".. Skipping existing results until label", skipped_workload.head()[0])
    return skipped_workload


def stop_evaluation(sig_num, stack_trace) -> None:
    global StopEvaluation
    StopEvaluation = True


def execute_query_handler(query: qal.SqlQuery, database: db.Database,
                          duration_sender: multiprocessing.connection.Connection) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    start_time = datetime.now()
    database.execute_query(query, cache_enabled=False)
    end_time = datetime.now()
    query_duration = (end_time - start_time).total_seconds()
    duration_sender.send(query_duration)


def generate_all_join_orders(query: qal.SqlQuery,
                             exhaustive_enumerator: enumeration.ExhaustiveJoinOrderGenerator) -> list[data.JoinTree]:
    exhaustive_join_order_generator = exhaustive_enumerator.all_join_orders_for(query)
    join_order_plans = []
    for i in range(ExhaustiveJoinOrderingLimit):
        try:
            next_join_order = next(exhaustive_join_order_generator)
            join_order_plans.append(next_join_order)
            if len(join_order_plans) % 100 == 0:
                print("... Generated", len(join_order_plans), "plans")
        except StopIteration:
            print("... All join orders generated,", len(join_order_plans), "total")
            break
    return join_order_plans


def generate_random_join_orders(query: qal.SqlQuery) -> list[data.JoinTree]:
    random_enumerator = enumeration.RandomJoinOrderGenerator()
    join_order_plans = []
    random_plan_hashes = set()
    current_plan_idx = 0
    random_join_order_generator = random_enumerator.random_join_order_for(query)
    while current_plan_idx < ExhaustiveJoinOrderingLimit:
        next_join_order = next(random_join_order_generator)
        next_hash = hash(next_join_order)
        if next_hash in random_plan_hashes:
            continue
        random_plan_hashes.add(next_hash)
        join_order_plans.append(next_join_order)
        current_plan_idx += 1
        if current_plan_idx % 100 == 0:
            print("...", len(join_order_plans), "plans sampled")
    return join_order_plans


@dataclasses.dataclass
class EvaluationResult:
    label: str
    query: qal.SqlQuery
    join_order: data.JoinTree
    execution_time: float


def execute_single_query(label: str, query: qal.SqlQuery, join_order: data.JoinTree, *, n_executed_plans: int = 0,
                         total_query_runtime: float = 0, db_system: systems.DatabaseSystem,
                         db_instance: db.Database) -> EvaluationResult:
    query_generator = db_system.query_adaptor()
    enforce_hash_join = operators.PhysicalOperatorAssignment(query)
    enforce_hash_join.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False)
    enforce_hash_join.set_operator_enabled_globally(operators.JoinOperators.SortMergeJoin, False)

    optimized_query = query_generator.adapt_query(query, join_order=join_order,
                                                  physical_operators=enforce_hash_join)

    query_timeout = (QuerySlowdownToleranceFactor * (total_query_runtime / n_executed_plans) if n_executed_plans
                     else DefaultQueryTimeout)
    query_duration_receiver, query_duration_sender = mp.Pipe(False)

    # TODO: what about query repetitions?
    query_execution_worker = mp.Process(target=execute_query_handler,
                                        args=(optimized_query, db_instance, query_duration_sender))
    query_execution_worker.start()

    query_execution_worker.join(query_timeout)
    if query_execution_worker.is_alive():
        query_execution_worker.terminate()
        query_execution_worker.join()
        query_runtime = np.inf
    else:
        query_runtime = query_duration_receiver.recv()
        total_query_runtime += query_runtime
        n_executed_plans += 1
    query_execution_worker.close()

    return EvaluationResult(label, query, join_order, query_runtime)


def evaluate_query(label: str, query: qal.SqlQuery, *, db_instance: db.Database,
                   db_system: systems.DatabaseSystem) -> Iterable[EvaluationResult]:
    print("... Building all plans")
    exhaustive_enumerator = enumeration.ExhaustiveJoinOrderGenerator()
    join_order_plans = generate_all_join_orders(query, exhaustive_enumerator)

    should_sample_randomly = False
    reached_exhaustion_limit = len(join_order_plans) == ExhaustiveJoinOrderingLimit
    if reached_exhaustion_limit:
        # Special case: there exist exactly as many join orders for the query as the ExhaustiveJoinOrderingLimit
        # If this situation occurs, we don't want to fall back to random sampling because this would mean we need
        # to sample all join orders randomly (which will probably take a _very_ long time). Instead, we just keep
        # all our plans
        try:
            next(exhaustive_enumerator.all_join_orders_for(query))
            should_sample_randomly = True
        except StopIteration:
            should_sample_randomly = False

    if should_sample_randomly:
        print("... Falling back to random sampling of join orders")
        join_order_plans = generate_random_join_orders(query)

    n_executed_plans = 0
    total_query_runtime = 0
    query_runtimes = []
    for join_order in join_order_plans:
        evaluation_result = execute_single_query(label, query, join_order, n_executed_plans=n_executed_plans,
                                                 total_query_runtime=total_query_runtime,
                                                 db_system=db_system, db_instance=db_instance)
        n_executed_plans += 1
        total_query_runtime += evaluation_result.execution_time
        query_runtimes.append(evaluation_result)

        if n_executed_plans % 100 == 0:
            print("...", n_executed_plans, "plans executed")

    print("... All plans executed")
    return query_runtimes


def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["join_order"] = df["join_order"].apply(data.JoinTree.as_list).apply(jsonize.to_json)
    return df


def main():
    signal.signal(signal.SIGINT, stop_evaluation)
    os.makedirs(OutputDirectory, exist_ok=True)
    pg_db = postgres.connect(config_file=".psycopg_connection_job")
    pg_system = systems.Postgres(pg_db)

    workload = workloads.job()
    workload = skip_existing_results(workload)

    for label, query in workload.entries():
        print(".. Now evaluating query", label)
        query_runtimes = evaluate_query(label, query, db_instance=pg_db, db_system=pg_system)

        result_df = pd.DataFrame(query_runtimes)
        out_file = OutputDirectory + OutputFileFormat.format(label=label)
        result_df = prepare_for_export(result_df)
        result_df.to_csv(out_file, index=False)

        if StopEvaluation:
            print(".. Ctl+C received, terminating evaluation")
            break


if __name__ == "__main__":
    main()
