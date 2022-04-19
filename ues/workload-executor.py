#!/usr/bin/env python3

import argparse
import json
import os
import random
from datetime import datetime

import pandas as pd
import psycopg2


SQL_COMMENT_PREFIX = "--"
DEFAULT_WORKLOAD_COL = "query"


def connect_postgres(parser: argparse.ArgumentParser, conn_str: str = None):
    if not conn_str:
        if not os.path.exists(".psycopg_connection"):
            parser.error("No connect string for psycopg given.")
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


def execute_query(query, workload_prefix: str, cursor: "psycopg2.cursor"):
    query_start = datetime.now()
    cursor.execute(query)
    query_end = datetime.now()

    query_res = cursor.fetchone()[0]
    if isinstance(query_res, dict):
        query_res = json.dumps(query_res)
    query_duration = query_end - query_start

    result_col = f"{workload_prefix}_result"
    runtime_col = f"{workload_prefix}_rt_total"
    return pd.Series({result_col: query_res, runtime_col: query_duration.total_seconds()})


def main():
    parser = argparse.ArgumentParser(description="Utility to run different SQL workloads on postgres instances.")
    parser.add_argument("input", action="store", help="File containing the workload")
    parser.add_argument("--out", "-o", action="store", default=None, help="Name of the output CSV file containing the workload results.")
    parser.add_argument("--csv", action="store_true", default=False, help="Parse input data as CSV file")
    parser.add_argument("--csv-col", action="store", default="query", help="In CSV mode, name of the column containing the queries")
    parser.add_argument("--pg-con", action="store", default="", help="Connect string to the postgres instance (psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")

    args = parser.parse_args()
    df_workload = read_workload_csv(args.input) if args.csv else read_workload_plain(args.input)
    workload_col = args.csv_col if args.csv else DEFAULT_WORKLOAD_COL

    pg_conn = connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()

    workload_res_df = df_workload[workload_col].apply(execute_query, workload_prefix=workload_col, cursor=pg_cursor)
    result_df = pd.merge(df_workload, workload_res_df, left_index=True, right_index=True, how="outer")

    out_file = args.out if args.out else generate_default_out_name()
    result_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
