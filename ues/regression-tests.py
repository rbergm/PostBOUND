#!/usr/bin/env python3

import sys

import pandas as pd
import psycopg2

conn = psycopg2.connect("dbname=imdb user=strix host=localhost")
cur = conn.cursor()

df_queries = pd.read_csv("transformed.csv")


def regression_test(query_row):
    label = query_row["label"]
    orig_query = query_row["query"]
    flattened_query = query_row["flattened_query"]

    try:
        cur.execute(orig_query)
        orig_res = cur.fetchone()[0]
    except psycopg2.ProgrammingError:
        print("### On label", label, "programming error")
        print("Original query was", orig_query)
        sys.exit(1)

    try:
        cur.execute(flattened_query)
        flattened_res = cur.fetchone()[0]
    except psycopg2.ProgrammingError:
        print("### On label", label, "programming error")
        print("Flattened query was", flattened_query)
        sys.exit(1)

    if orig_res != flattened_res:
        print("Regression found at query", label)
        print(".. Query is", orig_query)
        print(".. With cardinality", orig_res)
        print(".. Transformed query is", flattened_query)
        print(".. With cardinality", flattened_res)
        return True
    return False


df_queries["regression"] = df_queries.apply(regression_test, axis="columns")
if not df_queries.regression.any():
    print("No regressions found")
