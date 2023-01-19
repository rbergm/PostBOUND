#!/usr/bin/env python3

import argparse
import functools
import os
import sys
import warnings

import pandas as pd
import psycopg2


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def connect_postgres(parser: argparse.ArgumentParser, conn_str: str = None):
    if not conn_str:
        if not os.path.exists(".psycopg_connection"):
            parser.error("No connect string for psycopg given.")
        with open(".psycopg_connection", "r") as conn_file:
            conn_str = conn_file.readline().strip()
    conn = psycopg2.connect(conn_str)
    return conn


def execute_query(data, conn, cursor, *, col):
    query = data[col]
    label = data.get("label", "")
    try:
        cursor.execute(query)
        cardinality = cursor.fetchone()[0]
    except psycopg2.Error as err:
        cardinality = -1 * abs(hash(col))  # use hash(col) to identify the column, multiply by -1 to designate error
        conn.rollback()
        err_msg = (f"Received psycopg2 error on query with label {label} (index {data.name}): {err}" if label
                   else f"Received psycopg2 error on query {query} (index {data.name}): {err}")
        warnings.warn(err_msg)
    return cardinality


def main():
    parser = argparse.ArgumentParser(description="Utility ensure correctness of multiple variants of the same base "
                                     "query.")
    parser.add_argument("input", action="store", help="CSV file containing the input queries")
    parser.add_argument("--base", "-b", action="store", required=True, help="Column containing the base queries")
    parser.add_argument("--transformed", "-t", action="store", required=True, help="Column containing the transformed "
                        "queries")
    parser.add_argument("--out", "-o", action="store", default="", help="File write the results to")
    parser.add_argument("--pg-con", action="store", default="", help="Connect string to the postgres instance "
                        "(psycopg2 format). If omitted, the string will be read from the file .psycopg_connection")

    args = parser.parse_args()
    df = pd.read_csv(args.input)
    pg_conn = connect_postgres(parser, args.pg_con)
    pg_cursor = pg_conn.cursor()

    query_runner = functools.partial(execute_query, conn=pg_conn, cursor=pg_cursor)
    col_base_card = f"card_{args.base}"
    if col_base_card not in df.columns:
        eprint("Determining cardinalities for base queries")
        df[col_base_card] = df.apply(query_runner, col=args.base, axis="columns")
    else:
        eprint("Reusing existing cardinalities for base queries")

    eprint("Determining cardinalities for transformed queries")
    col_transformed_card = f"card_{args.transformed}"
    df[col_transformed_card] = df.apply(query_runner, col=args.transformed, axis="columns")

    df["passed"] = df[col_base_card] == df[col_transformed_card]
    regressions = not df["passed"].all()
    if regressions:
        eprint("Found regressions on", len(df[~df.passed]), "queries")
    else:
        eprint("No regressions were found")

    if args.out:
        df.to_csv(args.out, index=False)
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
