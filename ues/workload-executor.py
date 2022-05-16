#!/usr/bin/env python3

import argparse
import functools
import json
import os
import re
import signal
import sys
import warnings
from datetime import datetime
from typing import List

import pandas as pd
import psycopg2

SQL_COMMENT_PREFIX = "--"
DEFAULT_WORKLOAD_COL = "query"
EXPLAIN_PREFIX = "explain"
ANALYZE_PREFIX = re.compile(r"explain ((\(.*analyze.*\))|(analyze))")


def log(*args, **kwargs):
    print("..", *args, file=sys.stderr, **kwargs)


def dummy_logger(*args, **kwargs):
    return


def make_logger(logging_enabled: bool = True):
    return log if logging_enabled else dummy_logger


def exit_handler(sig, frame, logger=dummy_logger):
    logger("Ctl+C received, exiting")
    sys.exit(1)


class QueryMod:
    @staticmethod
    def parse(mod_str: str, error_reporter) -> "QueryMod":
        if mod_str == "explain":
            return QueryMod(explain=True)
        elif mod_str == "analyze":
            return QueryMod(analyze=True)
        elif mod_str == "":
            return QueryMod()
        else:
            error_reporter.error(f"Unkown query mod: '{mod_str}'")

    def __init__(self, explain: bool = False, analyze: bool = False):
        if explain and analyze:
            warnings.warn("Both explain and analyze given, but analyze subsumes explain. Ignoring additional explain.")
        self.explain = explain
        self.analyze = analyze

    def apply_mods(self, query):
        if self._needs_analyze(query):
            return f"explain (analyze, format json) {query}"
        elif self._needs_explain(query):
            return f"explain (format json) {query}"
        else:
            return query

    def _needs_explain(self, query):
        query = query.lower()
        return self.explain and not query.startswith(EXPLAIN_PREFIX)

    def _needs_analyze(self, query):
        query = query.lower()
        return self.analyze and not ANALYZE_PREFIX.match(query)


def connect_postgres(parser: argparse.ArgumentParser, conn_str: str = None):
    if not conn_str:
        if not os.path.exists(".psycopg_connection"):
            parser.error("No connect string for psycopg given and .psycopg_connection does not exist.")
        with open(".psycopg_connection", "r") as conn_file:
            conn_str = conn_file.readline().strip()
    conn = psycopg2.connect(conn_str)
    return conn


def generate_default_out_name(out_prefix="workload-out"):
    out_suffix = datetime.now().strftime("%y%m%d-%H%M")
    return f"{out_prefix}-{out_suffix}.csv"


def read_workload_plain(input: str) -> pd.DataFrame:
    workload = []
    with open(input, "r") as input_file:
        workload_raw = [query.strip() for query in input_file.readlines()]
        workload = [query for query in workload_raw if not query.startswith(SQL_COMMENT_PREFIX)]
    return pd.DataFrame({DEFAULT_WORKLOAD_COL: workload})


def read_workload_csv(input: str) -> pd.DataFrame:
    return pd.read_csv(input)


def execute_query(query, workload_prefix: str, cursor: "psycopg2.cursor", *,
                  pg_args: List[str], query_mod: QueryMod = None, query_hint: str = "", logger=dummy_logger):
    logger("Now running query", query)

    for arg in pg_args:
        logger("Applying postgres argument", arg)
        cursor.execute(arg)

    if query_mod is not None:
        orig_query = query
        query = query_mod.apply_mods(query)
        if orig_query != query:
            logger("Query modified to", query)

    if query_hint:
        logger("Applying query hint", query_hint)
        query = query_hint + " " + query

    query_start = datetime.now()
    cursor.execute(query)
    query_end = datetime.now()

    query_res = cursor.fetchone()[0]
    if isinstance(query_res, dict) or isinstance(query_res, list):
        query_res = json.dumps(query_res)
    query_duration = query_end - query_start

    logger("Query took", query_duration, "seconds")

    result_col = f"{workload_prefix}_result"
    runtime_col = f"{workload_prefix}_rt_total"
    return pd.Series({result_col: query_res, runtime_col: query_duration.total_seconds()})


def execute_query_wrapper(workload_row: pd.Series, workload_col: str, cursor: "psycopg2.cursor", *,
                          pg_args: List[str], query_mod: QueryMod = None, hint_col: str = "", logger=dummy_logger):
    if hint_col:
        hint = workload_row[hint_col]
    else:
        hint = None
    return execute_query(workload_row[workload_col], workload_prefix=workload_col, cursor=cursor,
                         pg_args=pg_args, query_mod=query_mod, query_hint=hint, logger=logger)


def run_workload(workload: pd.DataFrame, workload_col: str, cursor: "psycopg2.cursor", *,
                 pg_args: List[str], query_mod: QueryMod = None, hint_col: str = "", logger=dummy_logger):
    logger(len(workload), "queries total")
    workload_res_df = workload.apply(execute_query_wrapper,
                                     workload_col=workload_col, cursor=cursor,
                                     pg_args=pg_args, query_mod=query_mod, hint_col=hint_col,
                                     logger=logger,
                                     axis="columns")
    result_df = pd.merge(workload, workload_res_df, left_index=True, right_index=True, how="outer")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Utility to run different SQL workloads on postgres instances.")
    parser.add_argument("input", action="store", help="File containing the workload")
    parser.add_argument("--out", "-o", action="store", default=None, help="Name of the output CSV file containing "
                        "the workload results.")
    parser.add_argument("--csv", action="store_true", default=False, help="Parse input data as CSV file")
    parser.add_argument("--csv-col", action="store", default="query", help="In CSV mode, name of the column "
                        "containing the queries")
    parser.add_argument("--pg-con", action="store", default="", help="Connect string to the postgres instance "
                        "(psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")
    parser.add_argument("--pg-param", action="extend", default=[], type=str, nargs="*", help="Parameters to be send "
                        "to the Postgres instance")
    parser.add_argument("--query-mod", action="store", default="", help="Optional modifications of the base query. "
                        "Can be either 'explain' or 'analyze' to turn all queries into EXPLAIN or EXPLAIN ANALYZE "
                        "queries respectively.")
    parser.add_argument("--hint-col", action="store", default="", help="In CSV mode, an optional column containing "
                        "hints to apply on a per-query basis (as specified by the pg_hint_plan extension).")
    parser.add_argument("--verbose", action="store_true", default=False, help="Produce more debugging output")

    args = parser.parse_args()

    logger = make_logger(args.verbose)
    signal.signal(signal.SIGINT, functools.partial(exit_handler, logger=logger))

    df_workload = read_workload_csv(args.input) if args.csv else read_workload_plain(args.input)
    workload_col = args.csv_col if args.csv else DEFAULT_WORKLOAD_COL

    pg_conn = connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()

    query_mod = QueryMod.parse(args.query_mod, parser)

    result_df = run_workload(df_workload, workload_col, pg_cursor, pg_args=args.pg_param,
                             query_mod=query_mod, hint_col=args.hint_col,
                             logger=logger)

    out_file = args.out if args.out else generate_default_out_name()
    result_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
