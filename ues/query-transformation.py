#!/usr/bin/env python3

import collections
import pathlib
import functools
import sys

import mo_sql_parsing as mosp
import natsort
import numpy as np
import pandas as pd
import psycopg2

from transform import flatten, db

df_data = collections.defaultdict(list)
workload_path = pathlib.Path("../simplicity-done-right/JOB-Queries/explicit")
query_files = list(workload_path.glob("*.sql"))
df_data["label"].extend(query_file.stem for query_file in query_files)
df_data["query"].extend(query_file.read_text().replace("\n", " ").lower() for query_file in query_files)

conn = psycopg2.connect("dbname=imdb user=strix host=localhost")
cur = conn.cursor()
dbschema = db.DBSchema(cur)

df_queries = pd.DataFrame(df_data)
df_queries.sort_values(by="label", key=lambda _: np.argsort(natsort.index_natsorted(df_queries["label"])), inplace=True)
df_queries.reset_index(drop=True, inplace=True)

flattener = functools.partial(flatten.flatten_query, dbschema=dbschema)


def test(label):
    q = df_queries[df_queries.label == label].iloc[0]["query"]
    print(mosp.parse(q))
    print(flattener(q))

    sys.exit()


#test("11a")

df_queries["flattened_mosp"] = df_queries["query"].apply(flattener)


def flattened_formatter(df_row):
    label = df_row["label"]
    mosp_query = df_row["flattened_mosp"]
    flattened_sql = None
    try:
        flattened_sql = mosp.format(mosp_query)
    except:
        print("... Error on label", label)
        print("MOSP data is")
        print(mosp_query)
        print("=====")
    return flattened_sql


df_queries["flattened_query"] = df_queries.apply(flattened_formatter, axis="columns")
df_queries.to_csv("transformed.csv", columns=["label", "query", "flattened_query"], index=False)
