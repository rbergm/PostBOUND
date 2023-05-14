from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import multiprocessing.connection
import os
import pathlib
import re
import signal
from collections.abc import Sequence
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from postbound.db import db, postgres
from postbound.qal import qal, transform, parser
from postbound.experiments import workloads
from postbound.optimizer import jointree
from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import operators
from postbound.optimizer.planmeta import hints as params
from postbound.util import jsonize, misc

StopEvaluation = False
CancelEvaluation = False
Interactive = True
workloads.workloads_base_dir = "../workloads"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    exhaustive_join_ordering_limit: int = 500
    query_slowdown_tolerance_factor: int = 3
    min_queries_until_dynamic_timeout: int = 10
    default_query_timeout: int = 120
    minimum_query_timeout: int = 30
    timeout_mode: str = "native"  # allowed values: "native" / "dynamic"
    native_runtimes_df: str = "results/job/job-native-runtimes.csv"

    operator_selection: str = "native"  # allowed values: "native" / "hashjoin" / "optimal"
    enable_prewarming: bool = True
    true_cardinalities_df: str = "results/job/job-intermediate-cardinalities.csv"

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


def filter_for_label(workload: workloads.Workload, labels: Sequence[str]) -> workloads.Workload:
    if not len(labels):
        return workload
    if len(labels) == 1:
        target = labels[0]
        return workload.filter_by(lambda label, __: label == target)
    elif len(labels) == 2:
        start, end = labels
        return workload.filter_by(lambda label, __: start <= label <= end)
    else:
        allowed_labels = set(labels)
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
                             config: ExperimentConfig = ExperimentConfig.default()) -> list[jointree.LogicalJoinTree]:
    exhaustive_join_order_generator = exhaustive_enumerator.all_join_orders_for(query)
    join_order_plans = []
    for __ in range(config.exhaustive_join_ordering_limit):
        try:
            next_join_order = next(exhaustive_join_order_generator)
            join_order_plans.append(next_join_order)
            if len(join_order_plans) % 100 == 0:
                print("... Generated", len(join_order_plans), "plans")
        except StopIteration:
            print("... All join orders generated,", len(join_order_plans), "total")
            break
    return join_order_plans


def generate_random_join_orders(query: qal.SqlQuery, *, config: ExperimentConfig = ExperimentConfig.default()
                                ) -> list[jointree.LogicalJoinTree]:
    random_enumerator = enumeration.RandomJoinOrderGenerator()
    join_order_plans = []
    random_plan_hashes = set()
    current_plan_idx = 0
    random_join_order_generator = random_enumerator.random_join_order_generator(query)

    max_tries = 3 * config.exhaustive_join_ordering_limit
    current_try = 0
    while current_plan_idx < config.exhaustive_join_ordering_limit:
        current_try += 1
        if current_try > max_tries:
            print("... Stopping sampling since max tries have been reached")
            break

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
    join_order: jointree.JoinTree
    query_hints: str
    planner_options: str
    query_plan: dict
    cost_estimate: float
    execution_time: float
    timeout: float


def determine_timeout(label: str, total_query_runtime: float, n_executed_plans: int, *,
                      config: ExperimentConfig = ExperimentConfig.default()) -> float:
    if config.timeout_mode == "native":
        df = pd.read_csv(config.native_runtimes_df)
        native_runtime = df[df.label == label]["execution_time"].iloc[0]
        timeout = config.query_slowdown_tolerance_factor * native_runtime
        return max(timeout, config.minimum_query_timeout)

    if not config.timeout_mode == "dynamic":
        raise ValueError("Unknown timeout mode: " + config.timeout_mode)

    if n_executed_plans < config.min_queries_until_dynamic_timeout:
        return config.default_query_timeout
    average_runtime = total_query_runtime / n_executed_plans
    timeout = config.query_slowdown_tolerance_factor * average_runtime
    return max(timeout, config.minimum_query_timeout)


def restrict_to_hash_join() -> operators.PhysicalOperatorAssignment:
    operator_selection = operators.PhysicalOperatorAssignment()
    operator_selection.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False)
    operator_selection.set_operator_enabled_globally(operators.JoinOperators.SortMergeJoin, False)
    return operator_selection


def true_cardinality_hints(label: str, config: ExperimentConfig) -> params.PlanParameterization:
    card_df = pd.read_csv(config.true_cardinalities_df)
    relevant_queries = card_df[card_df.label == label].copy()
    relevant_queries["query_fragment"] = relevant_queries["query_fragment"].apply(parser.parse_query)
    plan_params = params.PlanParameterization()
    for __, row in relevant_queries.iterrows():
        plan_params.add_cardinality_hint(row["query_fragment"].tables(), row["cardinality"])
    return plan_params


def execute_single_query(label: str, query: qal.SqlQuery, join_order: jointree.LogicalJoinTree, *,
                         n_executed_plans: int = 0, total_query_runtime: float = 0,
                         db_instance: db.Database,
                         config: ExperimentConfig = ExperimentConfig.default()) -> EvaluationResult:
    query_generator = db_instance.hinting()

    if config.operator_selection == "native":
        operator_selection = None
        plan_params = None
    elif config.operator_selection == "hashjoin":
        operator_selection = restrict_to_hash_join()
        plan_params = None
    elif config.operator_selection == "optimal":
        operator_selection = None
        plan_params = true_cardinality_hints(label, config)
    else:
        raise ValueError("Unknown operator selection strategy: " + config.operator_selection)

    optimized_query = query_generator.generate_hints(query, join_order=join_order,
                                                     physical_operators=operator_selection,
                                                     plan_parameters=plan_params)

    query_timeout = determine_timeout(label, total_query_runtime, n_executed_plans, config=config)
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
    query_duration_receiver.close()
    query_duration_sender.close()

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


def evaluate_query(label: str, query: qal.SqlQuery, *, db_instance: db.Database,
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
                                                 db_instance=db_instance, config=config)
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
    df["join_order"] = df["join_order"].apply(jointree.JoinTree.as_list).apply(jsonize.to_json)
    df["query_plan"] = df["query_plan"].apply(jsonize.to_json)
    df["db_config"] = df["db_config"].apply(jsonize.to_json)
    return df


def read_config() -> tuple[ExperimentConfig, Sequence[str]]:
    if not Interactive:
        return ExperimentConfig.default(), ["1a"]

    arg_parser = argparse.ArgumentParser(description="Determines the difference in query runtime depending on the "
                                                     "selected join order for queries in the Join Order Benchmark.")

    arg_parser.add_argument("--max-plans", action="store", type=int, default=500,
                            help="The maximum number of join orders to evaluate per query.")
    arg_parser.add_argument("--slowdown-tolerance", action="store", type=float, default=2.5,
                            help="Maximum factor a query can be slower than the dynamic timeout before being "
                                 "cancelled.")
    arg_parser.add_argument("--default-timeout", action="store", type=int, default=120,
                            help="Default static query timeout.")
    arg_parser.add_argument("--min-timeout", action="store", type=int, default=30,
                            help="Minimum dynamic query timeout.")
    arg_parser.add_argument("--timeout-mode", action="store", default="dynamic", choices=["native", "dynamic"],
                            help="Calculate timeout based on current average runtime ('dynamic'), or based on the "
                                 "runtime of a natively optimized query ('native'). The tolerance factor is applied "
                                 "normally.")

    arg_parser.add_argument("--prewarm", action="store_true", default=False, help="Pre-warm the database buffer pool.")
    arg_parser.add_argument("--operator-selection", action="store", default="native",
                            choices=["native", "hashjoin", "optimal"],
                            help="Modify the operator selection. native for unchanged operators, hashjoin to "
                                 "restrict all joins or optimal to derive from true cardinalities.")

    arg_parser.add_argument("--out-dir", action="store", type=str, default="results/query-runtime-variation/",
                            help="Directory where to store the experiment results.")

    arg_parser.add_argument("--execute-all", action="store_true", default=False,
                            help="Force evaluation of all benchmark queries.")

    arg_parser.add_argument("labels", action="store", nargs="*",
                            help="Filters the workload for the given query labels: "
                                 "0 labels means all queries, "
                                 "1 label filters for exactly that query, "
                                 "2 labels filter for all queries between (and including) the given labels, "
                                 "3 or more labels again filter for exactly the given labels")

    args = arg_parser.parse_args()
    config = ExperimentConfig(exhaustive_join_ordering_limit=args.max_plans,
                              query_slowdown_tolerance_factor=args.slowdown_tolerance,
                              default_query_timeout=args.default_timeout,
                              minimum_query_timeout=args.min_timeout,
                              enable_prewarming=args.prewarm,
                              operator_selection=args.operator_selection,
                              output_directory=args.out_dir if args.out_dir.endswith("/") else args.out_dir + "/",
                              skip_existing_results=not args.execute_all)
    labels = list(args.labels)
    return config, labels


def main():
    signal.signal(signal.SIGINT, stop_evaluation)
    config, labels = read_config()
    os.makedirs(config.output_directory, exist_ok=True)
    pg_db = postgres.connect(config_file=".psycopg_connection_job")

    workload = workloads.job()

    if labels:
        print(".. Filtering workload for queries", labels)
        workload = filter_for_label(workload, labels)
    else:
        workload = skip_existing_results(workload, config=config)

    if CancelEvaluation:
        # in case evaluation was cancelled shortly after starting the experiment (and before running the first query),
        # exit here
        return

    for label, query in workload.entries():
        print(".. Now evaluating query", label, "time =", misc.current_timestamp())
        query_runtimes = evaluate_query(label, query, db_instance=pg_db, config=config)

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
