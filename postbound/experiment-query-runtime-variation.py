from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing as mp
import multiprocessing.connection
import os
import pathlib
import re
import signal
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from postbound.db import db, postgres
from postbound.qal import qal, base, transform
from postbound.experiments import workloads
from postbound.optimizer import jointree
from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import operators
from postbound.optimizer.planmeta import hints as params
from postbound.util import jsonize

StopEvaluation = False
CancelEvaluation = False
Interactive = True
logging_format = "%(asctime)s %(levelname)s %(message)s"
logging_level = logging.DEBUG
workloads.workloads_base_dir = "../workloads"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    exhaustive_join_ordering_limit: int = 1000
    query_slowdown_tolerance_factor: int = 3
    min_queries_until_dynamic_timeout: int = 10
    default_query_timeout: int = 120
    minimum_query_timeout: int = 15
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
    next_query = skipped_workload.head()
    assert next_query is not None
    logging.info("Skipping existing results until label %s", next_query[0])
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
    result_plan = database.execute_query(query, cache_enabled=False)
    end_time = datetime.now()
    query_duration = (end_time - start_time).total_seconds()
    duration_sender.send((result_plan, query_duration))


def generate_all_join_orders(query: qal.SqlQuery, exhaustive_enumerator: enumeration.ExhaustiveJoinOrderGenerator, *,
                             config: ExperimentConfig = ExperimentConfig.default()) -> list[jointree.LogicalJoinTree]:
    exhaustive_join_order_generator = exhaustive_enumerator.all_join_orders_for(query)
    join_order_plans = []
    for __ in range(config.exhaustive_join_ordering_limit):
        try:
            next_join_order = next(exhaustive_join_order_generator)
            join_order_plans.append(next_join_order)
            if len(join_order_plans) % 100 == 0:
                logging.debug("Generated %s plans", len(join_order_plans))
        except StopIteration:
            logging.debug("All join orders generated, %s total", len(join_order_plans))
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
            logging.debug("Stopping sampling since max tries have been reached")
            break

        next_join_order = next(random_join_order_generator)
        next_hash = hash(next_join_order)
        if next_hash in random_plan_hashes:
            continue

        random_plan_hashes.add(next_hash)
        join_order_plans.append(next_join_order)
        current_plan_idx += 1

        if current_plan_idx % 100 == 0:
            logging.debug("%s plans sampled", len(join_order_plans))

    return join_order_plans


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    label: str
    query: qal.SqlQuery
    join_order: jointree.JoinTree
    query_hints: str
    planner_options: str
    query_plan: dict
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


def assert_correct_query_plan(label: str, query: qal.SqlQuery, expected_join_order: jointree.JoinTree,
                              actual_query_plan: dict) -> None:
    parsed_actual_plan = postgres.PostgresExplainPlan(actual_query_plan).as_query_execution_plan()
    actual_join_order = jointree.LogicalJoinTree.load_from_query_plan(parsed_actual_plan, query)
    if expected_join_order != actual_join_order:
        logging.error("Join order was not enforced correctly for label %s", label)


OperatorSelection = tuple[Optional[operators.PhysicalOperatorAssignment], Optional[params.PlanParameterization]]


def native_operator_selection(join_order: jointree.JoinTree) -> OperatorSelection:
    plan_params = params.PlanParameterization()
    plan_params.set_system_settings(geqo="off")
    return None, plan_params


def restrict_to_hash_join(join_order: jointree.JoinTree) -> OperatorSelection:
    operator_selection = operators.PhysicalOperatorAssignment()
    operator_selection.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False)
    operator_selection.set_operator_enabled_globally(operators.JoinOperators.SortMergeJoin, False)

    plan_params = params.PlanParameterization()
    plan_params.set_system_settings(geqo="off")
    return operator_selection, plan_params


def parse_tables_list(tables: str) -> set[base.TableReference]:
    jsonized = json.loads(tables)
    return {base.TableReference(tab["full_name"], tab.get("alias")) for tab in jsonized}


class TrueCardinalityGenerator:
    def __init__(self, config: ExperimentConfig = ExperimentConfig.default()) -> None:
        self._card_df = pd.read_csv(config.true_cardinalities_df)
        self._card_df["tables"] = self._card_df["tables"].apply(parse_tables_list)
        self._current_label: Optional[str] = None
        self._relevant_queries: Optional[pd.DataFrame] = None

    def setup_for_query(self, label: str, query: qal.SqlQuery) -> None:
        self._current_label = label
        self._relevant_queries = self._card_df[self._card_df.label == label].copy()

    def __call__(self, join_order: jointree.JoinTree) -> OperatorSelection:
        assert self._relevant_queries is not None
        plan_params = params.PlanParameterization()
        for intermediate_join in join_order.join_sequence():
            joined_tables = intermediate_join.tables()
            current_cardinality = self._relevant_queries[self._relevant_queries.tables == joined_tables]
            if current_cardinality.empty:
                logging.warning("No cardinality found for intermediate %s at label %s", intermediate_join,
                                self._current_label)
                continue
            cardinality = current_cardinality.iloc[0]["cardinality"]
            plan_params.add_cardinality_hint(joined_tables, cardinality)
        for base_table in join_order.table_sequence():
            table = base_table.table
            current_cardinality = self._relevant_queries[self._relevant_queries.tables == {table}]
            if current_cardinality.empty:
                logging.warning("No cardinality found for base table %s at label %s", intermediate_join,
                                self._current_label)
                continue
            cardinality = current_cardinality.iloc[0]["cardinality"]
            plan_params.add_cardinality_hint([table], cardinality)

        plan_params.set_system_settings(geqo="off")
        return None, plan_params


def execute_single_query(label: str, query: qal.SqlQuery, join_order: jointree.LogicalJoinTree, *,
                         n_executed_plans: int = 0, total_query_runtime: float = 0,
                         db_instance: db.Database,
                         operator_generator: Callable[[jointree.JoinTree], OperatorSelection],
                         config: ExperimentConfig = ExperimentConfig.default()) -> EvaluationResult:
    query_generator = db_instance.hinting()

    operator_selection, plan_params = operator_generator(join_order)
    optimized_query = query_generator.generate_hints(query, join_order=join_order,
                                                     physical_operators=operator_selection,
                                                     plan_parameters=plan_params)
    optimized_query = transform.as_explain_analyze(optimized_query)

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

        # We cannot use db.optimizer().query_plan() here, b/c we need to JSON-serialize the raw plan later on. This is
        # currently not supported by the QueryExecutionPlan
        query_plan = db_instance.execute_query(transform.as_explain(optimized_query), cache_enabled=False)
        query_runtime = np.inf
    else:
        query_plan, query_runtime = query_duration_receiver.recv()
    query_execution_worker.close()
    query_duration_receiver.close()
    query_duration_sender.close()

    assert_correct_query_plan(label, query, join_order, query_plan)

    query_hints, planner_options = ((optimized_query.hints.query_hints, optimized_query.hints.preparatory_statements)
                                    if optimized_query.hints else ("", ""))
    return EvaluationResult(label=label, query=query, join_order=join_order,
                            query_hints=query_hints, planner_options=planner_options,
                            query_plan=query_plan, execution_time=query_runtime,
                            timeout=query_timeout)


def prewarm_database(query: qal.SqlQuery, db_instance: postgres.PostgresInterface, *,
                     config: ExperimentConfig = ExperimentConfig.default()) -> None:
    if not config.enable_prewarming:
        return
    db_instance.prewarm_tables(query.tables())


def evaluate_query(label: str, query: qal.SqlQuery, *, db_instance: postgres.PostgresInterface,
                   config: ExperimentConfig = ExperimentConfig.default()) -> Iterable[EvaluationResult]:
    logging.debug("Building all plans for query %s", label)
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
        logging.debug("Falling back to random sampling of join orders for query %s", label)
        join_order_plans = generate_random_join_orders(query, config=config)

    if config.operator_selection == "native":
        operator_generator = native_operator_selection
    elif config.operator_selection == "hashjoin":
        operator_generator = restrict_to_hash_join
    elif config.operator_selection == "optimal":
        operator_generator = TrueCardinalityGenerator(config)
        operator_generator.setup_for_query(label, query)
    else:
        raise ValueError("Unknown operator selection strategy: " + config.operator_selection)

    n_executed_plans = 0
    total_query_runtime = 0.0
    query_runtimes = []
    prewarm_database(query, db_instance, config=config)
    logging.debug("Starting query execution for query %s", label)
    for join_order in join_order_plans:
        evaluation_result = execute_single_query(label, query, join_order, n_executed_plans=n_executed_plans,
                                                 total_query_runtime=total_query_runtime,
                                                 db_instance=db_instance, operator_generator=operator_generator,
                                                 config=config)
        n_executed_plans += 1
        total_query_runtime += (evaluation_result.execution_time if math.isfinite(evaluation_result.execution_time)
                                else evaluation_result.timeout)
        query_runtimes.append(evaluation_result)

        if n_executed_plans % 100 == 0:
            logging.debug("%s plans executed", n_executed_plans)
        if CancelEvaluation:
            logging.info("Cancel")
            break

    logging.debug("All plans executed for query %s", label)
    return query_runtimes


def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["join_order"] = df["join_order"].apply(jointree.JoinTree.as_list).apply(jsonize.to_json)
    df["query_plan"] = df["query_plan"].apply(jsonize.to_json)
    df["db_config"] = df["db_config"].apply(jsonize.to_json)
    return df


def read_config() -> tuple[ExperimentConfig, Sequence[str]]:
    if not Interactive:
        logging.basicConfig(level=logging_level, format=logging_format, filename="experiment-test.log", filemode="w")
        console_logger = logging.StreamHandler()
        console_logger.setLevel(logging.DEBUG)
        console_logger.setFormatter(logging.Formatter(logging_format))
        logging.getLogger().addHandler(console_logger)
        return ExperimentConfig(operator_selection="optimal"), ["1a"]

    arg_parser = argparse.ArgumentParser(description="Determines the difference in query runtime depending on the "
                                                     "selected join order for queries in the Join Order Benchmark.")

    arg_parser.add_argument("--max-plans", action="store", type=int, default=1000,
                            help="The maximum number of join orders to evaluate per query.")
    arg_parser.add_argument("--slowdown-tolerance", action="store", type=float, default=2.5,
                            help="Maximum factor a query can be slower than the dynamic timeout before being "
                                 "cancelled.")
    arg_parser.add_argument("--default-timeout", action="store", type=int, default=120,
                            help="Default static query timeout.")
    arg_parser.add_argument("--min-timeout", action="store", type=int, default=15,
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
    arg_parser.add_argument("--log-file", "-l", action="store", type=str, default="query-runtime-variation.log",
                            help="File to write logging information to")

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
                              timeout_mode=args.timeout_mode,
                              enable_prewarming=args.prewarm,
                              operator_selection=args.operator_selection,
                              output_directory=args.out_dir if args.out_dir.endswith("/") else args.out_dir + "/",
                              skip_existing_results=not args.execute_all)
    labels = list(args.labels)

    logging.basicConfig(level=logging_level, format=logging_format, filename=args.log_file, filemode="w")
    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.DEBUG)
    console_logger.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(console_logger)

    return config, labels


def main():
    signal.signal(signal.SIGINT, stop_evaluation)
    config, labels = read_config()
    logging.debug("Experiment config finalized")
    os.makedirs(config.output_directory, exist_ok=True)
    pg_db = postgres.connect(config_file=".psycopg_connection_job")
    logging.debug("Obtained database connection")

    # HOTFIX: the Leading() hint of pg_hint_plan does not work if Postgres decides to use the GeQO
    # optimizer (likely b/c they only instrumented the Dynamic Programming-based optimizer)
    # Therefore, for all JOB queries with >= 12 tables, native optimization would take place!
    # To fix this, we need to disable the GeQO optimizer for our workload
    # A cleaner solution will likely be to integrate cleanup actions into the Hint clause and make
    # the Postgres interface insert a GeQO disable command before each optimized query is executed along
    # with a GeQO enable command after the query has been executed. Until then, this hotfix should work fine.
    logging.debug("Disabling GeQO")
    pg_db.execute_query("SET geqo = 'off';", cache_enabled=False)

    workload = workloads.job(simplified=False)
    logging.debug("Raw workload read")

    if labels:
        logging.debug("Filtering workload for queries %s", labels)
        workload = filter_for_label(workload, labels)
    else:
        workload = skip_existing_results(workload, config=config)

    if CancelEvaluation:
        # in case evaluation was cancelled shortly after starting the experiment (and before running the first query),
        # exit here
        return

    logging.info("Starting workload execution")
    for label, query in workload.entries():
        logging.info("Now evaluating query %s", label)
        query_runtimes = evaluate_query(label, query, db_instance=pg_db, config=config)

        result_df = pd.DataFrame(query_runtimes)
        result_df["db_config"] = [pg_db.describe()] * len(result_df)
        out_file = config.output_directory + config.output_file_format.format(label=label)
        result_df = prepare_for_export(result_df)
        result_df.to_csv(out_file, index=False)

        if StopEvaluation or CancelEvaluation:
            logging.info(".. Ctl+C received, terminating evaluation")
            break


if __name__ == "__main__":
    main()
