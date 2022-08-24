#!/usr/bin/env python3

import argparse
import json
import operator
import re
import warnings
from typing import Dict, FrozenSet, List, Union

import pandas as pd

from transform import db, mosp
from postgres import hint


# see https://regex101.com/r/HdKzQg/1
TableReferencePattern = re.compile(r"(?P<full_name>\S+) AS (?P<alias>\S+)")


def parse_table_list(raw_tables: List[str]) -> FrozenSet[db.TableRef]:
    parsed = []
    for raw_table in raw_tables:
        table_match = TableReferencePattern.match(raw_table)
        if not table_match:
            warnings.warn(f"Could not parse table reference '{raw_table}'")
            continue
        full_name, alias = table_match.group("full_name"), table_match.group("alias")
        parsed.append(db.TableRef(full_name, alias))
    return frozenset(parsed)


def parse_query_bounds(raw_bounds: List[Dict[str, Union[List[str], int]]]
                       ) -> Dict[FrozenSet[db.TableRef], hint.JoinBoundsData]:
    parsed = {}
    for bounds_dict in raw_bounds:
        tables = parse_table_list(bounds_dict["join"])
        parsed[tables] = hint.JoinBoundsData(bounds_dict["bound"], bounds_dict["input_bounds"])
    return parsed


def stringify_operator_bounds(bounds: Dict[FrozenSet[db.TableRef], Dict[str, int]]
                              ) -> List[Dict[str, Union[List[str], int]]]:
    stringified_bounds = []
    for join, operator_bounds in bounds.items():
        stringified_join = [str(table) for table in join]
        stringified = {
            "join": stringified_join,
            "ues_bound": operator_bounds["ues"],
            "nlj_bound": operator_bounds["nlj"],
            "hashjoin_bound": operator_bounds["hashjoin"],
            "mergejoin_bound": operator_bounds["mergejoin"]
        }
        stringified_bounds.append(stringified)
    return stringified_bounds


def read_input(src: str, query_col: str, *, bound_col: str = "", parse_bounds: bool = False) -> pd.DataFrame:
    df = pd.read_csv(src)
    df[query_col] = df[query_col].apply(mosp.MospQuery.parse)

    if parse_bounds:
        df[f"{bound_col}_internal"] = df[bound_col].apply(json.loads).apply(parse_query_bounds)

    return df


def idxnlj_subqueries(input_df: pd.DataFrame, query_col: str, hint_col: str, *,
                      nlj_scope: str, idxscan_target: str) -> pd.DataFrame:
    df = input_df.copy()
    df[hint_col] = (df[query_col]
                    .apply(hint.idxnlj_subqueries, nestloop=nlj_scope, idxscan=idxscan_target)
                    .apply(hint.HintedMospQuery.generate_sqlcomment, strip_empty=True))
    return df


def bound_hints(input_df: pd.DataFrame, query_col: str, bound_col: str, hint_col: str) -> pd.DataFrame:
    df = input_df.copy()
    df[hint_col] = (df
                    .apply(lambda query_row: hint.bound_hints(query_row[query_col],
                                                              query_row[f"{bound_col}_internal"]),
                           axis="columns")
                    .apply(hint.HintedMospQuery.generate_sqlcomment, strip_empty=True))
    return df


def operator_hints(input_df: pd.DataFrame, query_col: str, bound_col: str, hint_col: str, *,
                   indexlookup_penalty: float = hint.DEFAULT_IDXLOOKUP_PENALTY,
                   hashjoin_penalty: float = hint.DEFAULT_HASHJOIN_PENALTY,
                   verbose: bool = False) -> pd.DataFrame:
    df = input_df.copy()
    hinted_queries = df.apply(lambda query_row: hint.operator_hints(query_row[query_col],
                                                                    query_row[f"{bound_col}_internal"],
                                                                    indexlookup_penalty=indexlookup_penalty,
                                                                    hashjoin_penalty=hashjoin_penalty,
                                                                    verbose=verbose),
                              axis="columns")
    df[hint_col] = hinted_queries.apply(hint.HintedMospQuery.generate_sqlcomment, strip_empty=True)
    df["operator_bounds"] = (hinted_queries
                             .apply(operator.attrgetter("bounds_stats"))
                             .apply(stringify_operator_bounds)
                             .apply(json.dumps))
    return df


def main():
    parser = argparse.ArgumentParser(description="Generates SQL query hints for various workloads.")
    parser.add_argument("input", action="store", help="CSV file containing the workload queries")
    parser.add_argument("--mode", "-m", action="store", default="ues-idxnlj", choices=["ues-idxnlj", "ues-bounds",
                                                                                       "ues-operators"],
                        help="The kind of hints to produce. Mode 'ues-idxnlj' (the default) is supported  enforces an "
                        "Index-NestedLoopJoin in UES subqueries queries. Mode 'ues-bounds' adds cardinality bound "
                        "hints for all available joins. Mode 'ues-operators' adds operator hints based on upper "
                        "bounds.")
    parser.add_argument("--idx-target", action="store", default="fk", choices=["pk", "fk"], help="For "
                        "'ues-idxnlj'-mode: The subquery join-partner that should be implemented as IndexScan. Can be "
                        "either 'pk' or 'fk', denoting the Primary key table and Foreign key table, respectively.")
    parser.add_argument("--nlj-scope", action="store", default="first", choices=["first", "all"], help="For "
                        "'ues-idxnlj'-mode: How many Index-Nested loop joins should be generated. Can be either "
                        "'first' denoting only the innermost join, or 'all', denoting all joins.")
    parser.add_argument("--hashjoin-penalty", action="store", type=float, default=hint.DEFAULT_HASHJOIN_PENALTY,
                        help="For operator bounds mode: value in (0, 1] to penalize hash-joined tuples.")
    parser.add_argument("--idxlookup-penalty", action="store", type=float, default=hint.DEFAULT_IDXLOOKUP_PENALTY,
                        help="For operator bounds mode: value in (0, 1] to penalize each tuple in the index part of "
                        "an Index NLJ.")
    parser.add_argument("--bounds-col", action="store", default="ues_bounds", help="For bounds modes: the column"
                        "which contains the bound information. Should be formatted as produced by ues-generator.py.")
    parser.add_argument("--out", "-o", action="store", default="out.csv", help="Name of the CSV file to store the "
                        "output")
    parser.add_argument("--query-col", action="store", default="query", help="Name of the CSV column containing the "
                        "workload")
    parser.add_argument("--hint-col", action="store", default="hint", help="Name of the CSV column to write the "
                        "generated hints to")
    parser.add_argument("--verbose", action="store_true", default=False, help="Produce debugging output (actual "
                        "output depends on the mode).")

    args = parser.parse_args()
    bounds_modes = ["ues-bounds", "ues-operators"]
    df = read_input(args.input, args.query_col, parse_bounds=args.mode in bounds_modes, bound_col=args.bounds_col)

    if args.mode == "ues-idxnlj":
        df = idxnlj_subqueries(df, args.query_col, args.hint_col,
                               nlj_scope=args.nlj_scope, idxscan_target=args.idx_target)
    elif args.mode == "ues-bounds":
        df = bound_hints(df, args.query_col, args.bounds_col, args.hint_col)
    elif args.mode == "ues-operators":
        df = operator_hints(df, args.query_col, args.bounds_col, args.hint_col,
                            indexlookup_penalty=args.idxlookup_penalty, hashjoin_penalty=args.hashjoin_penalty,
                            verbose=args.verbose)

    if args.mode in bounds_modes:
        df.drop(columns=f"{args.bounds_col}_internal", inplace=True)

    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
