#!/usr/bin/env python3

import argparse
import collections
import os
import pathlib
import functools

import mo_sql_parsing as mosp
import natsort
import numpy as np
import pandas as pd
import psycopg2

from transform import flatten, db, util


def read_workload(directory: str) -> pd.DataFrame:
    df_data = collections.defaultdict(list)
    workload_path = pathlib.Path(directory)
    query_files = list(workload_path.glob("*.sql"))
    df_data["label"].extend(query_file.stem for query_file in query_files)
    df_data["query"].extend(query_file.read_text().replace("\n", " ").lower() for query_file in query_files)

    df_queries = pd.DataFrame(df_data)
    df_queries.sort_values(by="label", key=lambda _: np.argsort(natsort.index_natsorted(df_queries["label"])), inplace=True)
    df_queries.reset_index(drop=True, inplace=True)

    return df_queries


def transform_queries(df_queries: pd.DataFrame, dbschema: db.DBSchema) -> pd.DataFrame:
    df_queries = df_queries.copy()
    flattener = functools.partial(flatten.flatten_query, dbschema=dbschema)
    df_queries["mosp_linearized"] = df_queries["query"].apply(flattener)
    df_queries["query_linearized"] = df_queries.mosp_linearized.apply(mosp.format)
    return df_queries[["label", "query", "query_linearized"]].copy()


def main():
    parser = argparse.ArgumentParser(description="UES transformation tool to replace subqueries with linear join "
                                     "paths.")
    parser.add_argument("input", action="store",
                        help="Path to the file or directory which contains the input queries (depending on the mode)")
    parser.add_argument("--csv", action="store_true", default=False, help="Enters CSV-mode, reading a CSV file.")
    parser.add_argument("--out", "-o", action="store", required=True,
                        help="Output file to write the transformed queries to")
    parser.add_argument("--pg-con", action="store", default="", help="Connect string to the postgres instance "
                        "(psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")

    args = parser.parse_args()

    if not args.csv:
        parser.error("Only CSV mode is currently supported")

    df_queries = pd.read_csv(args.input)

    pg_conn = util.connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()
    dbschema = db.DBSchema(pg_cursor)

    df_transformed = transform_queries(df_queries, dbschema)
    df_transformed.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
