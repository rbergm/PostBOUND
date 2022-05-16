#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime

import pandas as pd

import importlib
executor = importlib.import_module("workload-executor")

DEFAULT_N_REPETITIONS = 3


def log(*args, **kwargs):
    print("..", *args, file=sys.stderr, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Utility to create reliable results from workloads in a "
                                     "reproducible manner.")
    parser.add_argument("input", action="store", help="File containing the workload to run")
    parser.add_argument("--csv", action="store_true", default=False, help="Parse input data as CSV file")
    parser.add_argument("--csv-col", action="store", default="query", help="In CSV mode, name of the column "
                        "containing the queries")
    parser.add_argument("--repetitions", "-r", action="store", default=DEFAULT_N_REPETITIONS, type=int, help="The "
                        "number of times the workload should be executed.")
    parser.add_argument("--out", "-o", action="store", help="Name of the output file to write the results to.")
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
    out_file = args.out if args.out else executor.generate_default_out_name("experiment")

    log(f"Running {args.repetitions} repetitions of workload from {args.input}.")
    log("Writing results to", args.out)
    log("Additional postgres argugments:", args.pg_param)

    log("Reading workload")
    workload = executor.read_workload_csv(args.input) if args.csv else executor.read_workload_plain(args.input)
    workload_col = args.csv_col if args.csv else executor.DEFAULT_WORKLOAD_COL

    log("Connecting to postgres")
    pg_conn = executor.connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()

    query_mod = executor.QueryMod.parse(args.query_mod, parser)

    workload_results = []
    for run in range(1, args.repetitions + 1):
        log("Starting workload iteration", run, "at", datetime.now().strftime("%y-%m-%d, %H:%M"))
        df_results = executor.run_workload(workload, workload_col, pg_cursor,
                                           pg_args=args.pg_param, query_mod=query_mod, hint_col=args.hint_col,
                                           logger=executor.make_logger(args.verbose))
        df_results["run"] = run
        workload_results.append(df_results)

    log("Exporting results")
    df_results = pd.concat(workload_results)
    df_results.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
