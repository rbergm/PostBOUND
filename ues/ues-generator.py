#!/usr/bin/env python3

import argparse
import json
import pathlib
import os
import textwrap
from datetime import datetime
from typing import Dict

import pandas as pd
import psycopg2

from transform import db, mosp, ues, util
from analysis import selection


DEFAULT_QUERY_COL = "query"
DEFAULT_UES_COL = "query_col"


def read_workload_raw(src_file: str) -> pd.DataFrame:
    with open(src_file, "r") as query_file:
        queries = [query for query in query_file.readlines() if not query.startswith("--")]
    return pd.DataFrame({DEFAULT_QUERY_COL: queries})


def read_workload_csv(src_file: str) -> pd.DataFrame:
    return pd.read_csv(src_file)


def read_workload_pattern(src_directory: str, pattern: str = "*.sql", *, load_labels: bool = False) -> pd.DataFrame:
    query_directory = pathlib.Path(src_directory)
    queries = []
    labels = []
    for src_file in query_directory.glob(pattern):
        with open(src_file, "r") as query_file:
            query = " ".join(query_file.readlines())
        queries.append(query)
        labels.append(src_file.stem)
    workload = pd.DataFrame({DEFAULT_QUERY_COL: queries})
    if load_labels:
        workload["label"] = labels
    return workload


def connect_postgres(parser: argparse.ArgumentParser, conn_str: str = None):
    if not conn_str:
        if not os.path.exists(".psycopg_connection"):
            parser.error("No connect string for psycopg given and .psycopg_connection does not exist.")
        with open(".psycopg_connection", "r") as conn_file:
            conn_str = conn_file.readline().strip()
    conn = psycopg2.connect(conn_str)
    return conn


def write_queries_csv(result_data: pd.DataFrame, out_fname: str = "ues-out.csv"):
    result_data.to_csv(out_fname, index=False)


def write_queries_stdout(result_data: pd.DataFrame, query_col: str = "query_ues"):
    for query in result_data[query_col]:
        print(f"{query};")


def jsonize_join_bounds(bounds: Dict[ues.JoinTree, int]) -> dict:
    jsonized_bounds = []
    for tree, bound in bounds.items():
        join_key = [str(tab) for tab in tree.all_tables()]
        jsonized = {"join": join_key, "bound": bound}
        jsonized_bounds.append(jsonized)
    return json.dumps(jsonized_bounds)


def optimize_workload(workload: pd.DataFrame, query_col: str, out_col: str, *,
                      table_estimation: str = "explain", join_estimation: str = "basic", subqueries: str = "defensive",
                      timing: bool = False, dbs: db.DBSchema = db.DBSchema.get_instance()) -> pd.DataFrame:
    logger = util.make_logger()
    optimized_queries = []
    optimization_time = []
    optimization_success = []
    intermediate_bounds = []
    parsed_queries = workload[query_col].apply(mosp.MospQuery.parse)

    for query_idx, query in enumerate(parsed_queries):
        optimization_start = datetime.now()
        try:
            optimized_query, query_bounds = ues.optimize_query(query,
                                                               table_cardinality_estimation=table_estimation,
                                                               join_cardinality_estimation=join_estimation,
                                                               subquery_generation=subqueries,
                                                               dbs=dbs, return_bounds=True)
            optimization_success.append(True)
            intermediate_bounds.append(jsonize_join_bounds(query_bounds))
        except Exception as e:
            optimized_query = query
            optimization_success.append(False)
            intermediate_bounds.append(None)
            query_text = workload["label"].iloc[query_idx] if "label" in workload else f"'{query}'"
            logger("Could not optimize query ", query_text, ": ", type(e).__name__, " (", e, ")", sep="")
        optimization_end = datetime.now()
        optimization_duration = optimization_end - optimization_start
        optimized_queries.append(optimized_query)
        optimization_time.append(optimization_duration.total_seconds())

    optimized_workload = workload.copy()
    optimized_workload[out_col] = optimized_queries
    optimized_workload["optimization_success"] = optimization_success
    optimized_workload["ues_bounds"] = intermediate_bounds
    if timing:
        optimized_workload["optimization_time"] = optimization_time
    return optimized_workload


def optimize_single(query: str, *, table_estimation: str = "explain", join_estimation: str = "basic",
                    subqueries: str = "defensive", exec: bool = False, dbs: db.DBSchema = db.DBSchema.get_instance()):
    parsed_query = mosp.MospQuery.parse(query)
    optimized_query = ues.optimize_query(parsed_query,
                                         table_cardinality_estimation=table_estimation,
                                         join_cardinality_estimation=join_estimation,
                                         subquery_generation=subqueries,
                                         dbs=dbs)
    db_connection = dbs.connection

    if not exec:
        # without execution we simply print the optimized query
        print(str(optimized_query))
    else:
        # with execution we need a bit more structure: output the optimized query as well as its execution time
        # separately
        print(".. Optimized query:")
        print(str(optimized_query))
        print(".. Executing query")
        cursor = db_connection.cursor()
        cursor.execute("set join_collapse_limit = 1; set enable_memoize = 'off'; set enable_nestloop = 'off';")
        exec_start = datetime.now()
        cursor.execute(str(optimized_query))
        exec_end = datetime.now()
        print(".. Query took", exec_end - exec_start, "seconds")


def main():
    description = """Optimizes SQL queries or batches of queries according to the UES algorithm.

    Incoming queries currently have to specify all joins inplicitly (i.e via SELECT * FROM R, S WHERE ...), explicit
    joins via the JOIN statement are not supported (yet). Furthermore, all joins have to be specified over a single
    join predicate to prevent unexpected behaviour. (i.e. SELECT * FROM R JOIN S on R.a = S.b AND R.c = S.d is not
    supported, yet). In some cases however, these queries might still work.

    [0] Hertzschuch et al.: Simplicity Done Right for Join Ordering. CIDR'21."""
    parser = argparse.ArgumentParser(description=textwrap.dedent(description),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", action="store", default="", help="Path to the input file in batch mode")
    parser.add_argument("--query", "-q", action="store_true", help="Treat the input as a literal SQL query and "
                        "optimize it. Always writes the optimized query stdout.")
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument("--csv", action="store_true", help="Read input queries from CSV file. Otherwise queries are "
                          "read on a per-line basis. The resulting CSV file will contain the same columns/entries as"
                          "the input file, plus additional output columns.")
    mode_grp.add_argument("--pattern", action="store", help="Instead of reading from a single file, treat the input "
                          "as a directory and read queries form all files matching the pattern (one query per file).")
    parser.add_argument("--query-col", action="store", default=DEFAULT_QUERY_COL, help="In CSV-mode, column "
                        "containing the raw query.")
    parser.add_argument("--timing", "-t", action="store_true", help="In (output) CSV-mode, also measure optimization"
                        "time. This setting is ignored if the results are not written to CSV.")
    parser.add_argument("--generate-labels", action="store_true", help="In (output) CSV-mode when loading from "
                        "pattern, use the file names as query labels.")
    parser.add_argument("--out-col", action="store", default=DEFAULT_UES_COL, help="In CSV-mode, name of the output "
                        "column (defaults to query_ues).")
    parser.add_argument("--table-estimation", action="store", choices=["explain", "sample"], default="explain",
                        help="How cardinalities of (filtered) base tables should be estimated. If 'explain', use the"
                        "Postgres internal optimizer (as obtained via EXPLAIN output). If 'sample', draw a 20 percent"
                        "sample of rows and count the result cardinality. Defaults to 'explain'.")
    parser.add_argument("--join-estimation", action="store", choices=["basic", "topk"], default="basic", help="How to"
                        "estimate the upper bound of join cardinalities. If 'basic', use the Most frequent value as "
                        "detailed in the fundamental paper [0]. If 'topk', use the Top-K lists as detailed in the "
                        "Diploma thesis. Defaults to 'basic'.")
    parser.add_argument("--subqueries", action="store", choices=["greedy", "disabled", "defensive"],
                        default="defensive", help="When to pull PK/FK joins into subqueries. If 'greedy' all "
                        "possible PK/FK joins will be executed as subqueries. If 'disabled', never filter via "
                        "subqueries. If 'defensive' (as described in [0]), only generate subqueries if an improvement "
                        "is guaranteed. Defaults to 'defensive'.")
    parser.add_argument("--out", "-o", action="store", help="Enter output CSV-mode and store the output in file "
                        "rather than writing to stdout.")
    parser.add_argument("--exec", "-e", action="store_true", default=False, help="If optimizing a single query, also "
                        "execute it.")
    parser.add_argument("--pg-con", action="store", default="", help="Connect string to the Postgres instance "
                        "(psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")

    args = parser.parse_args()
    db_connection = connect_postgres(args.pg_con)
    dbs = db.DBSchema.get_instance(db_connection)

    # if the query param is given, we switch to single optimization mode: we only optimize this one query
    # in this case, we might also run the query online
    if args.query:
        optimize_single(args.input, exec=args.exec,
                        table_estimation=args.table_estimation,
                        join_estimation=args.join_estimation,
                        subqueries=args.subqueries, dbs=dbs)
        return

    # otherwise we need to read our workload depending on the input mode
    if args.csv:
        workload = read_workload_csv(args.input)
    elif args.pattern:
        workload = read_workload_pattern(args.input, args.pattern, load_labels=args.generate_labels)
    else:
        workload = read_workload_raw(args.input)

    if args.generate_labels:
        workload = selection.reorder(workload)

    optimized_workload = optimize_workload(workload, args.query_col, args.out_col,
                                           table_estimation=args.table_estimation,
                                           join_estimation=args.join_estimation,
                                           subqueries=args.subqueries,
                                           timing=args.timing,
                                           dbs=dbs)

    if args.out:
        write_queries_csv(optimized_workload, args.out)
    else:
        write_queries_stdout(optimized_workload, args.out_col)


if __name__ == "__main__":
    main()
