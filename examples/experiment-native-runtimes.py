from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from postbound import postgres, transform, workloads
from postbound.util import jsonize

logging_format = "%(asctime)s %(levelname)s %(message)s"
logging_level = logging.DEBUG


@dataclass
class BenchmarkResult:
    label: str
    query: str
    execution_time: float
    query_plan: str
    db_config: str


def execute_single_iteration(
    workload: workloads.Workload,
    *,
    database: postgres.PostgresInterface,
    prewarm: bool = False,
) -> pd.DataFrame:
    benchmark_results: list[BenchmarkResult] = []
    db_config = database.describe()

    for label, query in workload.entries():
        if prewarm:
            database.prewarm_tables(query.tables())
        query_start = datetime.now()
        database.execute_query(query)
        query_end = datetime.now()
        execution_time = (query_end - query_start).total_seconds()
        query_plan = database.execute_query(transform.as_explain_analyze(query))
        result_wrapper = BenchmarkResult(
            label,
            str(query),
            execution_time,
            jsonize.to_json(query_plan),
            jsonize.to_json(db_config),
        )
        benchmark_results.append(result_wrapper)

    return pd.DataFrame(benchmark_results).reset_index()


def execute_multiple_iterations(
    workload: workloads.Workload,
    *,
    database: postgres.PostgresInterface,
    repetitions: int = 1,
    prewarm: bool = False,
    shuffled: bool = False,
) -> pd.DataFrame:
    iteration_dfs: list[pd.DataFrame] = []
    n_queries = len(workload)

    for i in range(repetitions):
        logging.debug("Starting iteration %s", i)
        if shuffled:
            workload = workload.shuffle()

        current_df = execute_single_iteration(
            workload, database=database, prewarm=prewarm
        )
        execution_offset = i * n_queries
        current_df.insert(0, "execution_index", current_df["index"] + execution_offset)
        current_df.insert(1, "repetition", i + 1)
        current_df.drop(columns="index", inplace=True)

        iteration_dfs.append(current_df)

    return pd.concat(iteration_dfs)


def make_out_name(args: argparse.Namespace) -> str:
    if args.out:
        return args.out
    elif args.repetitions > 1:
        return "job-native-runtime-variation.csv"
    else:
        return "job-native-runtimes.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Obtains execution times for different benchmarks"
    )
    parser.add_argument(
        "--bench",
        "-b",
        action="store",
        default="job",
        choices=["job", "stats"],
        help="The benchmark to execute.",
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        action="store",
        type=int,
        default=1,
        help="The number of repetitions per query.",
    )
    parser.add_argument(
        "--shuffle",
        "-s",
        action="store_true",
        help="When using query repetitions, shuffle the workload "
        "after each workload iteration. If shuffling is not used, all repetitions are "
        "executed in sequence.",
    )
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="Try to pre-warm the buffer pool before executing each query.",
    )
    parser.add_argument(
        "--out",
        "-o",
        action="store",
        default="",
        help="Name and location of the output CSV file to create.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging_level, format=logging_format)

    pg_conf = (
        ".psycopg_connection_job"
        if args.bench == "job"
        else ".psycopg_connection_stats"
    )
    postgres_db = postgres.connect(config_file=pg_conf)
    postgres_db.cache_enabled = False
    workload = workloads.job() if args.bench == "job" else workloads.stats()

    if args.repetitions == 1:
        result_df = execute_single_iteration(
            workload, database=postgres_db, prewarm=args.prewarm
        )
    else:
        result_df = execute_multiple_iterations(
            workload,
            database=postgres_db,
            repetitions=args.repetitions,
            prewarm=args.prewarm,
            shuffled=args.shuffle,
        )

    out_file = make_out_name(args)
    result_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
