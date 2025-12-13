#!/usr/bin/env python3

import argparse
import collections
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import postbound as pb


@dataclass
class QueryIntermediate:
    label: str
    full_query: pb.SqlQuery
    query_fragment: pb.SqlQuery


def iter_intermediates(
    workload: pb.workloads.Workload,
) -> Generator[QueryIntermediate, None, None]:
    for label, query in workload.entries():
        if not isinstance(query, pb.qal.ImplicitSqlQuery):
            warnings.warn(
                f"Skipping query {label} b/c query fragments can currently "
                "only be created for implicit queries."
            )
            continue

        tables = query.tables()
        query_predicates = query.predicates()
        candidate_joins = pb.util.powerset(tables)

        for joined_tables in candidate_joins:
            if not joined_tables:
                continue
            if not query_predicates.joins_tables(joined_tables):
                continue

            joined_tables = sorted(joined_tables)
            query_fragment = pb.transform.extract_query_fragment(query, joined_tables)
            assert query_fragment is not None
            query_fragment = pb.transform.as_count_star_query(query_fragment)
            yield QueryIntermediate(label, query, query_fragment)


def simulate_intermediate_generation(out_file: str, workload: pb.workloads.Workload):
    unique_intermediates = set()
    intermediates_per_query: dict[pb.SqlQuery, int] = collections.defaultdict(int)
    for intermediate in iter_intermediates(workload):
        unique_intermediates.add(intermediate.query_fragment)
        intermediates_per_query[intermediate.full_query] += 1

    result_df = pd.DataFrame(
        intermediates_per_query.items(), columns=["query", "n_intermediates"]
    )
    result_df.to_csv(out_file, index=False)
    print(
        "unique",
        len(unique_intermediates),
        "total",
        sum(intermediates_per_query.values()),
    )


def determine_intermediates(
    benchmark: pb.workloads.Workload[str],
    *,
    out_file: str,
    pg_conf: str = ".psycopg_connection",
    timeout: Optional[int] = None,
    simulate_only: bool = False,
) -> None:
    postgres_db = pb.postgres.connect(config_file=pg_conf)

    if simulate_only:
        simulate_intermediate_generation(out_file, benchmark)
        return

    db_pool = pb.postgres.ParallelQueryExecutor(
        postgres_db.connect_string, n_threads=12, timeout=timeout
    )

    explored_queries: set[pb.SqlQuery] = set()
    fragment_to_queries_map: dict[pb.SqlQuery, list[pb.SqlQuery]] = (
        collections.defaultdict(list)
    )
    n_queries = 0

    for intermediate in iter_intermediates(benchmark):
        fragment_to_queries_map[intermediate.query_fragment].append(
            intermediate.full_query
        )
        if intermediate.query_fragment in explored_queries:
            continue
        explored_queries.add(intermediate.query_fragment)
        db_pool.queue_query(intermediate.query_fragment)
        n_queries += 1

    print(f".. All queries submitted to DB pool - {n_queries} total")
    db_pool.drain_queue()
    fragment_cardinalities: dict[pb.SqlQuery, int] = db_pool.result_set()

    queries, fragments, labels, fragment_tables, cardinalities = [], [], [], [], []
    for result_fragment, cardinality in fragment_cardinalities.items():
        for query in fragment_to_queries_map[result_fragment]:
            query_label = benchmark.label_of(query)
            queries.append(query)
            fragments.append(result_fragment)
            labels.append(query_label)
            fragment_tables.append(pb.util.to_json(list(result_fragment.tables())))
            cardinalities.append(cardinality)

    result_df = pd.DataFrame(
        {
            "label": labels,
            "query": queries,
            "query_fragment": fragments,
            "tables": fragment_tables,
            "cardinality": cardinalities,
        }
    )
    result_df = pb.bench.sort_results(result_df, "label")
    result_df.to_csv(out_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Calculates the cardinalities of all possible intermediate results of common "
        "benchmarks"
    )
    parser.add_argument(
        "--bench",
        "-b",
        action="store",
        choices=["job", "stats"],
        help="The benchmark to estimate",
    )
    parser.add_argument(
        "--pg-conf",
        "-c",
        action="store",
        help="Path to the Postgres connection config file",
    )
    parser.add_argument(
        "--out", "-o", action="store", help="Name and location of the output CSV file"
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Don't actually calculate the intermediates. Instead, determine "
        "which intermediates would be calculated",
    )
    parser.add_argument(
        "--timeout", action="store", type=int, help="Timeout for each query in seconds"
    )
    args = parser.parse_args()

    benchmark = pb.workloads.job() if args.bench == "job" else pb.workloads.stats()
    determine_intermediates(
        benchmark,
        pg_conf=args.pg_conf,
        out_file=args.out,
        timeout=args.timeout,
        simulate_only=args.dry,
    )


if __name__ == "__main__":
    main()
