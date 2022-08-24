#!/usr/bin/env python3

import argparse
import functools
import re
import signal
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

from transform import util

import importlib
executor = importlib.import_module("workload-executor")

DEFAULT_N_REPETITIONS = 3

# see https://regex101.com/r/Wzsdjq/1
POSTGRES_PARAM_PATTERN = re.compile(r"SET (?P<param>\S+)\s?=\s?(?P<value>\S+);?", flags=re.IGNORECASE)


def evaluate_result(result_df: pd.DataFrame, query_col: str, *, exclusive: bool = True):
    if not exclusive:
        print("Results:")
    print(result_df.groupby("run")[f"{query_col}_rt_total"].sum())


def parse_postgres_params(params: List[str]) -> Dict[str, str]:
    param_dict = {}
    for param in params:
        param = param.strip()

        # ignore Postgres comments
        if not param or param.startswith("--"):
            continue

        param_match = POSTGRES_PARAM_PATTERN.match(param)
        if not param_match:
            util.print_stderr(f"WARN: Could not parse postgres parameter '{param}', ignoring.")
            continue

        param_name, param_value = param_match.group("param"), param_match.group("value")
        if param_name in param_dict:
            util.print_stderr(f"WARN: Postgres parameter '{param_name}' has already been specified. "
                              f"Overwriting with {param_value}")

        param_dict[param_name] = param if param.strip().endswith(";") else param + ";"
    return param_dict


def main():
    parser = argparse.ArgumentParser(description="Utility to create reliable results from workloads in a "
                                     "reproducible manner.")
    parser.add_argument("input", action="store", nargs="+", help="File containing the workload to run")
    parser.add_argument("--csv", action="store_true", default=False, help="Parse input data as CSV file")
    parser.add_argument("--csv-col", action="store", default="query", help="In CSV mode or eval mode, name of the "
                        "column containing the queries")
    parser.add_argument("--repetitions", "-r", action="store", default=DEFAULT_N_REPETITIONS, type=int, help="The "
                        "number of times the workload should be executed.")
    parser.add_argument("--out", "-o", action="store", help="Name of the output file to write the results to.")
    parser.add_argument("--pg-con", action="store", default="", help="Connect string to the postgres instance "
                        "(psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")
    parser.add_argument("--pg-param", action="extend", default=[], type=str, nargs="*", help="Parameters to be send "
                        "to the Postgres instance. These take precedence over all other parameter settings.")
    parser.add_argument("--load-params", action="store", type=str, help="File containing additional Postgres "
                        "parameters (one per line). These take precedence over --experiment-mode parameters.")
    parser.add_argument("--experiment-mode", action="store", choices=["none", "ues"], default="non", type=str,
                        help="Load a pre-defined set of Postgres parameters to apply to each query.")
    parser.add_argument("--query-mod", action="store", default="", help="Optional modifications of the base query. "
                        "Can be either 'explain' or 'analyze' to turn all queries into EXPLAIN or EXPLAIN ANALYZE "
                        "queries respectively.")
    parser.add_argument("--hint-col", action="store", default="", help="In CSV mode, an optional column containing "
                        "hints to apply on a per-query basis (as specified by the pg_hint_plan extension).")
    parser.add_argument("--eval", action="store_true", default=False, help="Don't run a new experiment, but evaluate "
                        "the results given by the input result file.")
    parser.add_argument("--randomized", action="store_true", default=False, help="Execute the queries in a random "
                        "order.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Don't actually run any experiments, "
                        "just show which settings would be applied.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Produce progress output")
    parser.add_argument("--trace", action="store_true", default=False, help="Produce more debugging output")

    args = parser.parse_args()
    verbose = args.verbose or args.trace or args.dry_run
    log = util.make_logger(verbose, file=sys.stdout)
    log("Invocation:", " ".join(['"{}"'.format(arg) if isinstance(arg, str) and " " in arg else arg
                                 for arg in sys.argv]))
    signal.signal(signal.SIGINT, functools.partial(executor.exit_handler, logger=log))

    if args.eval:
        files_to_analyze = util.enlist(args.input)
        for input_file in files_to_analyze:
            if util.contains_multiple(args.input):
                print(f"===> {input_file} <===")
            result_df = pd.read_csv(input_file)
            query_col = args.csv_col if args.csv_col else executor.DEFAULT_WORKLOAD_COL
            evaluate_result(result_df, query_col, exclusive=True)
            if util.contains_multiple(args.input):
                print()
        return
    args.input = util.simplify(args.input)

    out_file = args.out if args.out else executor.generate_default_out_name("experiment")

    postgres_params: Dict[str, str] = {}
    if args.experiment_mode == "ues":
        raw_params = ["SET enable_nestloop = 'off';",
                      "SET join_collapse_limit = 1;",
                      "SET enable_memoize = 'off';"]
        postgres_params = util.dict_merge(postgres_params, parse_postgres_params(raw_params))
    if args.load_params:
        with open(args.load_params, "r") as param_file:
            raw_params = param_file.readlines()
            postgres_params = util.dict_merge(postgres_params, parse_postgres_params(raw_params))
    if args.pg_param:
        raw_params = args.pg_param
        postgres_params = util.dict_merge(postgres_params, parse_postgres_params(raw_params))
    postgres_params = list(postgres_params.values())

    log(f"Running {args.repetitions} repetitions of workload from {args.input}.")
    log("Writing results to", args.out)
    log("Additional postgres argugments:", postgres_params)

    log("Reading workload")
    workload = executor.read_workload_csv(args.input) if args.csv else executor.read_workload_plain(args.input)
    workload_col = args.csv_col if args.csv else executor.DEFAULT_WORKLOAD_COL

    log("Connecting to postgres")
    pg_conn = executor.connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()

    query_mod = executor.QueryMod.parse(args.query_mod, parser)

    if args.dry_run:
        log("Dry run finished.")
        return

    workload_results = []
    for run in range(1, args.repetitions + 1):
        log("Starting workload iteration", run, "at", datetime.now().strftime("%y-%m-%d, %H:%M"))
        df_results = executor.run_workload(workload, workload_col, pg_cursor,
                                           pg_args=postgres_params, query_mod=query_mod, hint_col=args.hint_col,
                                           shuffle=args.randomized,
                                           logger=executor.make_logger(args.trace))
        df_results["run"] = run
        workload_results.append(df_results)

    log("Exporting results")
    df_results = pd.concat(workload_results)
    df_results.to_csv(out_file, index=False)

    if verbose:
        evaluate_result(df_results, workload_col, exclusive=False)


if __name__ == "__main__":
    main()
