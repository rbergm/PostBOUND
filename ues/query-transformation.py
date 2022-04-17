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

from transform import flatten, db


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


def connect_postgres(parser: argparse.ArgumentParser, conn_str: str = None):
    if not conn_str:
        if not os.path.exists(".psycopg_connection"):
            parser.error("No connect string for psycopg given.")
        with open(".psycopg_connection", "r") as conn_file:
            conn_str = conn_file.readline().strip()
    conn = psycopg2.connect(conn_str)
    return conn


def transform_queries(df_queries: pd.DataFrame, dbschema: db.DBSchema) -> pd.DataFrame:
    df_queries = df_queries.copy()
    flattener = functools.partial(flatten.flatten_query, dbschema=dbschema)
    df_queries["flattened_mosp"] = df_queries["query"].apply(flattener)
    df_queries["flattened_query"] = df_queries.flattened_mosp.apply(mosp.format)
    return df_queries[["label", "query", "flattened_query"]]


def main():
    parser = argparse.ArgumentParser(description="UES transformation tool to replace subqueries with linear join paths.")
    parser.add_argument("--workload", "-w", action="store", required=True, help="Path to the directory which contains the input queries (one query per file)")
    parser.add_argument("--out", "-o", action="store", required=True, help="Name of the output file containing the transformed queries")
    parser.add_argument("--pg-con", action="store", default="", help="Connect string to the postgres instance (psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")

    args = parser.parse_args()
    df_queries = read_workload(args.workload)
    pg_conn = connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()
    dbschema = db.DBSchema(pg_cursor)

    df_transformed = transform_queries(df_queries, dbschema)
    df_transformed.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
