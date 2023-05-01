from __future__ import annotations

import dataclasses
import multiprocessing as mp
import multiprocessing.connection
import os
import pathlib
import re
import signal
import sys
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from postbound.db import db, postgres
from postbound.db.systems import systems
from postbound.qal import qal, transform
from postbound.experiments import workloads
from postbound.optimizer import data
from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import operators
from postbound.util import jsonize, misc

StopEvaluation = False
CancelEvaluation = False
workloads.workloads_base_dir = "../workloads"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    exhaustive_join_ordering_limit: int = 500
    query_slowdown_tolerance_factor: int = 3
    min_queries_until_dynamic_timeout: int = 10
    default_query_timeout: int = 120
    minimum_query_timeout: int = 30

    restricted_operator_selection: bool = True
    enable_prewarming: bool = True

    skip_existing_results: bool = True
    output_file_format: str = "join-order-runtimes-{label}.csv"
    output_file_pattern: str = r"join-order-runtimes-(?P<label>\w+).csv"
    output_file_glob: str = "join-order-runtimes-*.csv"
    output_directory: str = "results/query-runtime-variation/"

    @staticmethod
    def default() -> ExperimentConfig:
        return ExperimentConfig()


def skip_existing_results(workload: workloads.Workload, *,
                          config: ExperimentConfig = ExperimentConfig.default()) -> workloads.Workload:
    if not config.skip_existing_results:
        return workload
    existing_results = set()
    result_directory = pathlib.Path(config.output_directory)
    for result_file in result_directory.glob(config.output_file_glob):
        file_matcher = re.match(config.output_file_pattern, result_file.name)
        if not file_matcher:
            continue
        label = file_matcher.group("label")
        if not label:
            continue
        existing_results.add(label)

    if not existing_results:
        return workload
    skipped_workload = workload.filter_by(lambda l, __: l not in existing_results)
    print(".. Skipping existing results until label", skipped_workload.head()[0])
    return skipped_workload


def filter_for_label(workload: workloads.Workload) -> workloads.Workload:
    if len(sys.argv) < 2:
        return workload
    if len(sys.argv) == 2:
        target = sys.argv[1]
        return workload.filter_by(lambda label, __: label == target)
    elif len(sys.argv) == 3:
        start, end = sys.argv[1:]
        return workload.filter_by(lambda label, __: start <= label <= end)
    else:
        allowed_labels = set(sys.argv[1:])
        return workload.filter_by(lambda label, __: label in allowed_labels)


def stop_evaluation(sig_num, stack_trace) -> None:
    global StopEvaluation
    global CancelEvaluation
    if StopEvaluation:  # 2nd time the signal is raised cancel the entire evaluation
        CancelEvaluation = True
    StopEvaluation = True  # don't evaluate any new queries if the signal is raised


def execute_query_handler(query: qal.SqlQuery, database: db.Database,
                          duration_sender: multiprocessing.connection.Connection) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    start_time = datetime.now()
    database.execute_query(query, cache_enabled=False)
    end_time = datetime.now()
    query_duration = (end_time - start_time).total_seconds()
    duration_sender.send(query_duration)


def generate_all_join_orders(query: qal.SqlQuery, exhaustive_enumerator: enumeration.ExhaustiveJoinOrderGenerator, *,
                             config: ExperimentConfig = ExperimentConfig.default()) -> list[data.JoinTree]:
    exhaustive_join_order_generator = exhaustive_enumerator.all_join_orders_for(query)
    join_order_plans = []
    for i in range(config.exhaustive_join_ordering_limit):
        try:
            next_join_order = next(exhaustive_join_order_generator)
            join_order_plans.append(next_join_order)
            if len(join_order_plans) % 100 == 0:
                print("... Generated", len(join_order_plans), "plans")
        except StopIteration:
            print("... All join orders generated,", len(join_order_plans), "total")
            break
    return join_order_plans


def generate_random_join_orders(query: qal.SqlQuery, *,
                                config: ExperimentConfig = ExperimentConfig.default()) -> list[data.JoinTree]:
    random_enumerator = enumeration.RandomJoinOrderGenerator()
    join_order_plans = []
    random_plan_hashes = set()
    current_plan_idx = 0
    random_join_order_generator = random_enumerator.random_join_order_generator(query)
    while current_plan_idx < config.exhaustive_join_ordering_limit:
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


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    label: str
    query: qal.SqlQuery
    join_order: data.JoinTree
    query_hints: str
    planner_options: str
    query_plan: dict
    cost_estimate: float
    execution_time: float
    timeout: float


def determine_timeout(total_query_runtime: float, n_executed_plans: int, *,
                      config: ExperimentConfig = ExperimentConfig.default()) -> float:
    if n_executed_plans < config.min_queries_until_dynamic_timeout:
        return config.default_query_timeout
    average_runtime = total_query_runtime / n_executed_plans
    timeout = config.query_slowdown_tolerance_factor * average_runtime
    return max(timeout, config.minimum_query_timeout)


def restrict_to_hash_join(query: qal.SqlQuery) -> operators.PhysicalOperatorAssignment:
    operator_selection = operators.PhysicalOperatorAssignment(query)
    operator_selection.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False)
    operator_selection.set_operator_enabled_globally(operators.JoinOperators.SortMergeJoin, False)
    return operator_selection


def execute_single_query(label: str, query: qal.SqlQuery, join_order: data.JoinTree, *, n_executed_plans: int = 0,
                         total_query_runtime: float = 0, db_system: systems.DatabaseSystem, db_instance: db.Database,
                         config: ExperimentConfig = ExperimentConfig.default()) -> EvaluationResult:
    query_generator = db_system.query_adaptor()
    operator_selection = restrict_to_hash_join(query) if config.restricted_operator_selection else None

    optimized_query = query_generator.adapt_query(query, join_order=join_order,
                                                  physical_operators=operator_selection)

    query_timeout = determine_timeout(total_query_runtime, n_executed_plans, config=config)
    query_duration_receiver, query_duration_sender = mp.Pipe(False)

    # TODO: what about query repetitions?
    query_execution_worker = mp.Process(target=execute_query_handler,
                                        args=(optimized_query, db_instance, query_duration_sender))
    query_execution_worker.start()

    query_execution_worker.join(query_timeout)
    timed_out = query_execution_worker.is_alive()
    if timed_out:
        query_execution_worker.terminate()
        query_execution_worker.join()
        db_instance.reset_connection()

        query_runtime = np.inf
    else:
        query_runtime = query_duration_receiver.recv()
    query_execution_worker.close()

    query_plan = db_instance.execute_query(transform.as_explain(optimized_query), cache_enabled=False)
    cost_estimate = db_instance.cost_estimate(optimized_query)
    query_hints, planner_options = ((optimized_query.hints.query_hints, optimized_query.hints.preparatory_statements)
                                    if optimized_query.hints else ("", ""))
    return EvaluationResult(label=label, query=query, join_order=join_order,
                            query_hints=query_hints, planner_options=planner_options,
                            query_plan=query_plan, cost_estimate=cost_estimate, execution_time=query_runtime,
                            timeout=query_timeout)


def prewarm_database(query: qal.SqlQuery, db_instance: db.Database, *,
                     config: ExperimentConfig = ExperimentConfig.default()) -> None:
    if not config.enable_prewarming:
        return
    for table in query.tables():
        db_instance.execute_query(f"SELECT pg_prewarm('{table.full_name}');", cache_enabled=False)


def evaluate_query(label: str, query: qal.SqlQuery, *, db_instance: db.Database, db_system: systems.DatabaseSystem,
                   config: ExperimentConfig = ExperimentConfig.default()) -> Iterable[EvaluationResult]:
    print("... Building all plans")
    exhaustive_enumerator = enumeration.ExhaustiveJoinOrderGenerator()
    join_order_plans = generate_all_join_orders(query, exhaustive_enumerator, config=config)

    should_sample_randomly = False
    reached_exhaustion_limit = len(join_order_plans) == config.exhaustive_join_ordering_limit
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
        join_order_plans = generate_random_join_orders(query, config=config)

    n_executed_plans = 0
    total_query_runtime = 0
    query_runtimes = []
    prewarm_database(query, db_instance, config=config)
    print("... Starting query execution")
    for join_order in join_order_plans:
        evaluation_result = execute_single_query(label, query, join_order, n_executed_plans=n_executed_plans,
                                                 total_query_runtime=total_query_runtime,
                                                 db_system=db_system, db_instance=db_instance,
                                                 config=config)
        n_executed_plans += 1
        total_query_runtime += (evaluation_result.execution_time if np.isfinite(evaluation_result.execution_time)
                                else evaluation_result.timeout)
        query_runtimes.append(evaluation_result)

        if n_executed_plans % 100 == 0:
            print("...", n_executed_plans, "plans executed")
        if CancelEvaluation:
            print("... Cancel")
            break

    print("... All plans executed")
    return query_runtimes


def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["join_order"] = df["join_order"].apply(data.JoinTree.as_list).apply(jsonize.to_json)
    df["query_plan"] = df["query_plan"].apply(jsonize.to_json)
    df["db_config"] = df["db_config"].apply(jsonize.to_json)
    return df


def main():
    signal.signal(signal.SIGINT, stop_evaluation)
    config = ExperimentConfig.default()
    os.makedirs(config.output_directory, exist_ok=True)
    pg_db = postgres.connect(config_file=".psycopg_connection_job")
    pg_system = systems.Postgres(pg_db)

    workload = workloads.job()

    if len(sys.argv) > 1:
        print(".. Filtering workload for queries", sys.argv[1:])
        workload = filter_for_label(workload)
    else:
        workload = skip_existing_results(workload, config=config)

    if CancelEvaluation:
        # in case evaluation was cancelled shortly after starting the experiment (and before running the first query),
        # exit here
        return

    for label, query in workload.entries():
        print(".. Now evaluating query", label, "time =", misc.current_timestamp())
        query_runtimes = evaluate_query(label, query, db_instance=pg_db, db_system=pg_system, config=config)

        result_df = pd.DataFrame(query_runtimes)
        result_df["db_config"] = [pg_db.inspect()] * len(result_df)
        out_file = config.output_directory + config.output_file_format.format(label=label)
        result_df = prepare_for_export(result_df)
        result_df.to_csv(out_file, index=False)

        if StopEvaluation or CancelEvaluation:
            print(".. Ctl+C received, terminating evaluation")
            break


if __name__ == "__main__":
    main()
