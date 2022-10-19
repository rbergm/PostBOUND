#!/usr/bin/env python3

import argparse
import functools
import json
import random
import os
import re
import signal
import sys
import warnings
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import psycopg2

from transform import util

SQL_COMMENT_PREFIX = "--"
DEFAULT_WORKLOAD_COL = "query"
EXPLAIN_PREFIX = "explain"
ANALYZE_PREFIX = re.compile(r"explain ((\(.*analyze.*\))|(analyze))")


executed_queries_counter = util.AtomicInt()


def log(*args, **kwargs):
    print("..", *args, file=sys.stderr, **kwargs)


def dummy_logger(*args, **kwargs):
    return


def make_logger(logging_enabled: bool = True):
    return log if logging_enabled else dummy_logger


def time():
    return datetime.now().strftime("%y-%m-%d, %H:%M")


def exit_handler(sig, frame, logger=dummy_logger):
    logger("\nCtl+C received, exiting")
    sys.exit(1)


def progress_logger(log, total_queries):
    global executed_queries_counter

    def _logger():
        global executed_queries_counter
        executed_queries_counter.increment()
        current_value = executed_queries_counter.value
        if current_value % (total_queries // 10) == 0 and current_value:
            log(f"Now executing query {current_value} of {total_queries} at {time()}")
    return _logger


class QueryMod:
    @staticmethod
    def parse(mod_str: str, error_reporter) -> "QueryMod":
        explain = "explain" in mod_str
        analyze = "analyze" in mod_str
        count_star = "count" in mod_str

        if not explain and not analyze and not count_star and mod_str != "":
            error_reporter.error(f"Unkown query mod: '{mod_str}'")

        return QueryMod(explain=explain, analyze=analyze, count_star=count_star)

    def __init__(self, *, explain: bool = False, analyze: bool = False, count_star: bool = False):
        if explain and analyze:
            warnings.warn("Both explain and analyze given, but analyze subsumes explain. Ignoring additional explain.")
        self.explain = explain
        self.analyze = analyze
        self.count_star = count_star

    def apply_mods(self, query):
        if self.count_star:
            query = self._apply_count_star(query)

        if self._needs_analyze(query):
            return f"explain (analyze, format json) {query}"
        elif self._needs_explain(query):
            return f"explain (format json) {query}"
        else:
            return query

    def _needs_explain(self, query: str) -> bool:
        query = query.lower()
        return self.explain and not query.startswith(EXPLAIN_PREFIX)

    def _needs_analyze(self, query: str) -> bool:
        query = query.lower()
        return self.analyze and not ANALYZE_PREFIX.match(query)

    def _apply_count_star(self, query: str) -> str:
        normalized_query = query.lower()

        if "having" in query:
            warnings.warn("Cannot transform a query with HAVING clause to SELECT COUNT(*), leaving query unmodified")
            return query

        select_idx = normalized_query.find("select")
        from_idx = normalized_query.find("from")

        select_clause = "SELECT COUNT(*) "
        current_select_clause = query[select_idx:from_idx]
        modified_query = query.replace(current_select_clause, select_clause, 1)

        # drop ORDER BY and GROUP BY since it is now meaningless and messes things up it the ordering was derived from
        # an aggregated attribut specified in the (now gone) SELECT clause
        normalized_query = modified_query.lower()
        grp_idx = normalized_query.find("group by")
        if grp_idx > 0:
            modified_query = modified_query[:grp_idx]
        order_idx = normalized_query.find("order by")
        if order_idx > 0:
            modified_query = modified_query[:order_idx]

        return modified_query

    def __repr__(self) -> str:
        str(self)

    def __str__(self) -> str:
        return f"QueryMod (explain={self.explain}, analyze={self.analyze}, count={self.count_star})"


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
    with open(input, "r", encoding="utf-8") as input_file:
        workload_raw = [query.strip() for query in input_file.readlines()]
        workload = [query for query in workload_raw if not query.startswith(SQL_COMMENT_PREFIX)]
    return pd.DataFrame({DEFAULT_WORKLOAD_COL: workload})


def read_workload_csv(input: str) -> pd.DataFrame:
    return pd.read_csv(input)


def shuffle_workload(df):
    shuffled_idx = random.sample(df.index.values.tolist(), k=len(df))
    return df.loc[shuffled_idx].copy()


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

    if isinstance(query_hint, str) and query_hint:
        logger("Applying query hint", query_hint)
        query = query_hint + " " + query

    query_start = datetime.now()
    cursor.execute(query)
    query_end = datetime.now()

    raw_res = cursor.fetchall()
    if len(raw_res) == 1:
        query_res = raw_res[0]
        complex_type = isinstance(query_res, dict) or isinstance(query_res, list)
        multiple_attrs = isinstance(query_res, tuple) and len(query_res) > 1
        if complex_type or multiple_attrs:
            query_res = json.dumps(query_res)
        else:
            query_res = query_res[0]
    elif raw_res:
        query_res = json.dumps(raw_res)
    else:
        query_res = None
    query_duration = query_end - query_start

    logger("Query took", query_duration, "seconds")

    result_col = f"{workload_prefix}_result"
    runtime_col = f"{workload_prefix}_rt_total"
    return pd.Series({result_col: query_res, runtime_col: query_duration.total_seconds()})


def execute_query_wrapper(workload_row: pd.Series, workload_col: str, cursor: "psycopg2.cursor", *,
                          pg_args: List[str], query_mod: QueryMod = None, hint_col: str = "",
                          progress_logger=None, logger=dummy_logger):
    if hint_col:
        hint = workload_row[hint_col]
    else:
        hint = None
    if progress_logger:
        progress_logger()
    return execute_query(workload_row[workload_col], workload_prefix=workload_col, cursor=cursor,
                         pg_args=pg_args, query_mod=query_mod, query_hint=hint, logger=logger)


def run_workload(workload: pd.DataFrame, workload_col: str, cursor: "psycopg2.cursor", *,
                 pg_args: List[str], query_mod: QueryMod = None, hint_col: str = "", shuffle: bool = False,
                 log_progress: bool = False, logger=dummy_logger):
    global executed_queries_counter

    logger(len(workload), "queries total")
    log_progress = progress_logger(log, len(workload)) if log_progress else progress_logger(dummy_logger, np.inf)
    executed_queries_counter.reset()

    if shuffle:
        workload = shuffle_workload(workload)

    workload_res_df = workload.apply(execute_query_wrapper,
                                     workload_col=workload_col, cursor=cursor,
                                     pg_args=pg_args, query_mod=query_mod, hint_col=hint_col,
                                     logger=logger, progress_logger=log_progress,
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
    parser.add_argument("--query-mod", action="store", default="", help="Optional (comma separated) modifications of "
                        "the base query. Can be either 'explain' or 'analyze' to turn all queries into EXPLAIN or "
                        "EXPLAIN ANALYZE queries respectively, or 'count' to execute all queries as COUNT (*) "
                        "instead of their normal projection.")
    parser.add_argument("--hint-col", action="store", default="", help="In CSV mode, an optional column containing "
                        "hints to apply on a per-query basis (as specified by the pg_hint_plan extension).")
    parser.add_argument("--randomized", action="store_true", default=False, help="Execute the queries in a random "
                        "order.")
    parser.add_argument("--log-progress", action="store_true", default=False, help="Write a progress message to "
                        "stdout every 1/10th of the total workload.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Produce more debugging output")
    parser.add_argument("--trace", action="store_true", default=False, help="Produce even more debugging output")

    args = parser.parse_args()

    logger = make_logger(args.verbose or args.trace)
    signal.signal(signal.SIGINT, functools.partial(exit_handler, logger=logger))

    df_workload = read_workload_csv(args.input) if args.csv else read_workload_plain(args.input)
    workload_col = args.csv_col if args.csv else DEFAULT_WORKLOAD_COL

    pg_conn = connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()

    query_mod = QueryMod.parse(args.query_mod, parser)

    result_df = run_workload(df_workload, workload_col, pg_cursor, pg_args=args.pg_param,
                             query_mod=query_mod, hint_col=args.hint_col,
                             shuffle=args.randomized,
                             log_progress=args.log_progress, logger=logger)

    out_file = args.out if args.out else generate_default_out_name()
    result_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
