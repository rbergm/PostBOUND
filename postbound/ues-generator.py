#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import pprint
import sys
import textwrap
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import psycopg2

from transform import db, mosp, ues, util
from analysis import selection


DEFAULT_QUERY_COL = "query"
DEFAULT_UES_COL = "query_ues"

PG_ARGS = {"join_collapse_limit": 1, "enable_nestloop": "off"}
PG_ARGS_13 = util.dict_merge(PG_ARGS, {"enable_memoize": "off"})


def read_workload_raw(src_file: str) -> pd.DataFrame:
    with open(src_file, "r", encoding="utf-8") as query_file:
        queries = [query for query in query_file.readlines() if not query.startswith("--")]
    return pd.DataFrame({DEFAULT_QUERY_COL: queries})


def read_workload_csv(src_file: str) -> pd.DataFrame:
    return pd.read_csv(src_file)


def read_workload_pattern(src_directory: str, pattern: str = "*.sql", *, load_labels: bool = False) -> pd.DataFrame:
    query_directory = pathlib.Path(src_directory)
    queries = []
    labels = []
    for src_file in query_directory.glob(pattern):
        with open(src_file, "r", encoding="utf-8") as query_file:
            query = " ".join(line.strip() for line in query_file.readlines() if not line.lstrip().startswith("--"))
        queries.append(query)
        labels.append(src_file.stem)
    workload = pd.DataFrame({DEFAULT_QUERY_COL: queries})
    if load_labels:
        workload["label"] = labels
    return workload


def parse_exception_list(path: str) -> List[dict]:
    with open(path, "r") as exceptions_file:
        raw_exception_content = json.load(exceptions_file)

    rules = []
    for raw_exception_rule in util.enlist(raw_exception_content):
        label = raw_exception_rule.get("label", "")
        query = raw_exception_rule.get("query", "")
        subquery_generation = raw_exception_rule.get("subquery_generation", True)
        if query:
            rules.append({"query": query, "subquery_generation": subquery_generation})
        else:
            rules.append({"label": label, "subquery_generation": subquery_generation})

    return rules


def resolve_exception_rule_labels(exception_rules: List[dict], workload: pd.DataFrame,
                                  query_col: str) -> ues.ExceptionList:
    resolved_rules = []

    for rule in exception_rules:
        if "query" in rule or "label" not in workload:
            query = rule["query"]
            resolved_rules.append((query,
                                   ues.ExceptionRule(query=query, subquery_generation=rule["subquery_generation"])))
        else:
            resolved_query = workload[workload.label == rule["label"]].iloc[0][query_col]
            resolved_rules.append((resolved_query, ues.ExceptionRule(rule["label"], resolved_query,
                                                                     rule["subquery_generation"])))

    return ues.ExceptionList(dict(resolved_rules))


def pg_connect_str(parser: argparse.ArgumentParser, conn_str: str = None) -> str:
    if not conn_str:
        if not os.path.exists(".psycopg_connection"):
            parser.error("No connect string for psycopg given and .psycopg_connection does not exist.")
        with open(".psycopg_connection", "r") as conn_file:
            conn_str = conn_file.readline().strip()
    return conn_str


def write_queries_csv(result_data: pd.DataFrame, out_fname: str = "ues-out.csv"):
    result_data.to_csv(out_fname, index=False)


def write_queries_stdout(result_data: pd.DataFrame, query_col: str = "query_ues"):
    for query in result_data[query_col]:
        print(f"{query};")


def optimize_workload(workload: pd.DataFrame, query_col: str, out_col: str, *,
                      table_estimation: str = "explain", join_estimation: str = "basic", subqueries: str = "defensive",
                      topk_length: int = None, timing: bool = False, exceptions: ues.ExceptionList = None,
                      verbose: bool = False, trace: bool = False,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> pd.DataFrame:
    logger = util.make_logger()
    optimized_queries = []
    optimization_time = []
    optimization_success = []
    intermediate_bounds = []
    final_bounds = []
    parsed_queries = workload[query_col].apply(mosp.MospQuery.parse)

    for query_idx, query in enumerate(parsed_queries):
        optimization_start = datetime.now()
        try:
            opt_res: ues.OptimizationResult = ues.optimize_query(query,
                                                                 table_cardinality_estimation=table_estimation,
                                                                 join_cardinality_estimation=join_estimation,
                                                                 subquery_generation=subqueries,
                                                                 topk_list_length=topk_length, exceptions=exceptions,
                                                                 verbose=trace, introspective=True,
                                                                 dbs=dbs)
            optimized_query, query_bounds = opt_res.query, opt_res.bounds
            intermediate_bounds.append(util.to_json(query_bounds))
            final_bounds.append(opt_res.final_bound)
            optimization_success.append(True)
        except Exception as e:
            optimized_query = query
            optimization_success.append(False)
            intermediate_bounds.append([])
            final_bounds.append(np.nan)
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
    optimized_workload["ues_final_bound"] = final_bounds
    if timing:
        optimized_workload["optimization_time"] = optimization_time
    return optimized_workload


def optimize_single(query: str, *, table_estimation: str = "explain", join_estimation: str = "basic",
                    subqueries: str = "defensive", topk_length: int = None,
                    exec: bool = False, print_join_path: bool = False, print_bounds: bool = False,
                    exceptions: ues.ExceptionList, verbose: bool = False, trace: bool = False,
                    dbs: db.DBSchema = db.DBSchema.get_instance()) -> None:
    parsed_query = mosp.MospQuery.parse(query)
    optimization_result: ues.OptimizationResult = ues.optimize_query(parsed_query,
                                                                     table_cardinality_estimation=table_estimation,
                                                                     join_cardinality_estimation=join_estimation,
                                                                     subquery_generation=subqueries,
                                                                     topk_list_length=topk_length,
                                                                     exceptions=exceptions,
                                                                     verbose=verbose, trace=trace,
                                                                     introspective=True,
                                                                     dbs=dbs)
    optimized_query = optimization_result.query

    structured_output_mode = exec or print_join_path or print_bounds
    if not structured_output_mode:
        # without special output, we simply print the optimized query
        print(str(optimized_query))
        return

    # otherwise, we generate additional output
    print()
    print(".. Optimized query:")
    print(str(optimized_query))

    if print_join_path:
        print()
        print(".. Final join path:")
        print(optimized_query.join_path(short=True))

    if print_bounds:
        print()
        print(".. Calculated bounds:")
        pprint.pprint({join_path: bounds["bound"] for join_path, bounds in optimization_result.bounds.items()})

    if exec:
        print()
        print(".. Executing query")
        pg_args = PG_ARGS_13 if dbs.pg_version() > "12" else PG_ARGS
        exec_start = datetime.now()
        dbs.execute_query(str(optimized_query), cache_enabled=False,
                          **pg_args)
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
    parser.add_argument("--join-paths", action="store_true", help="In (output) CSV mode, also store the final join "
                        "path per query. In single optimization mode the join path will be printed alongside the "
                        "optimized query. Otherwise this setting is ignored.")
    parser.add_argument("--out-col", action="store", default=DEFAULT_UES_COL, help="In CSV-mode, name of the output "
                        "column (defaults to query_ues).")
    parser.add_argument("--table-estimation", action="store", choices=["explain", "sample", "precise"],
                        default="explain", help="How cardinalities of (filtered) base tables should be estimated. If "
                        "'explain', use the Postgres internal optimizer (as obtained via EXPLAIN output). If "
                        "'sample', draw a 20 percent sample of rows and count the result cardinality. If 'precise', "
                        "actually execute the filter predicate and count the result tuples. Defaults to 'explain'.")
    parser.add_argument("--join-estimation", action="store", choices=["basic", "topk", "topk-approx"], default="basic",
                        help="How to estimate the upper bound of join cardinalities. If 'basic', use the Most "
                        "frequent value as detailed in the fundamental paper [0]. If 'topk', use the Top-K lists. If "
                        "'topk-approx', also leverage the Top-k lists, but in the approximative formula. Defaults to "
                        "'basic'.")
    parser.add_argument("--subqueries", action="store", choices=["greedy", "disabled", "defensive", "smart"],
                        default="defensive", help="When to pull PK/FK joins into subqueries. If 'greedy' all "
                        "possible PK/FK joins will be executed as subqueries. If 'disabled', never filter via "
                        "subqueries. If 'defensive' (as described in [0]), only generate subqueries if an improvement "
                        "is guaranteed. If 'smart', only generate subqueries if a \"worthwhile\" improvement can be "
                        "achieved (which is left intentionally ambiguous). Defaults to 'defensive'.")
    parser.add_argument("--topk-length", action="store", type=int, default=None, help="For Top-k join estimation, the"
                        "size of the MCV list (i.e. the k parameter).")
    parser.add_argument("--exception-list", action="store", help="JSON-File containing exceptions from the default"
                        "optimization settings.")
    parser.add_argument("--out", "-o", action="store", help="Enter output CSV-mode and store the output in file "
                        "rather than writing to stdout.")
    parser.add_argument("--exec", "-e", action="store_true", default=False, help="If optimizing a single query, also "
                        "execute it.")
    parser.add_argument("--bounds", action="store_true", default=False, help="If optimizing a single query, print "
                        "the resulting bounds.")
    parser.add_argument("--pg-con", action="store", default=None, help="Connect string to the Postgres instance "
                        "(psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print debugging output for generator.")
    parser.add_argument("--trace", action="store_true", default=False, help="Generate more debugging output for "
                        "optimization.")

    args = parser.parse_args()
    db_connect_str = pg_connect_str(parser, args.pg_con)
    dbs = db.DBSchema.get_instance(db_connect_str, renew=True)

    exceptions = parse_exception_list(args.exception_list) if args.exception_list else None

    # if the query param is given, we switch to single optimization mode: we only optimize this one query
    # in this case, we might also run the query online
    if args.query:
        optimize_single(args.input, exec=args.exec, print_join_path=args.join_paths, print_bounds=args.bounds,
                        table_estimation=args.table_estimation,
                        join_estimation=args.join_estimation,
                        topk_length=args.topk_length,
                        subqueries=args.subqueries,
                        exceptions=exceptions,
                        verbose=args.verbose, trace=args.trace, dbs=dbs)
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

    if args.exception_list:
        exceptions = resolve_exception_rule_labels(exceptions, workload, args.query_col)
        if args.verbose:
            rule_printing = [ues.ExceptionRule(label=rule.label, subquery_generation=rule.subquery_generation)
                             for rule in exceptions.rules.values()]
            util.print_stderr("Exception rules:")
            pprint.pprint(rule_printing, stream=sys.stderr)

    optimized_workload = optimize_workload(workload, args.query_col, args.out_col,
                                           table_estimation=args.table_estimation,
                                           join_estimation=args.join_estimation,
                                           subqueries=args.subqueries,
                                           topk_length=args.topk_length,
                                           exceptions=exceptions,
                                           timing=args.timing,
                                           verbose=args.verbose, trace=args.trace, dbs=dbs)

    if args.join_paths:
        optimized_workload["ues_join_path"] = optimized_workload[args.out_col].apply(mosp.MospQuery.join_path,
                                                                                     short=True)

    if args.out:
        write_queries_csv(optimized_workload, args.out)
    else:
        write_queries_stdout(optimized_workload, args.out_col)


if __name__ == "__main__":
    main()
