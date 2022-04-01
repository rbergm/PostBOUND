#!/usr/bin/env python3
import collections
from typing import Any, Dict

import pandas as pd
import mo_sql_parsing as mosp

# get access to our utility scripts
import sys
utils_path = "../utils/"
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
from helper import unwrap, select
import smat

df = pd.read_csv("../pg-bao/workloads/job-ues-cout.csv")
df["sql"] = df["query"].apply(mosp.parse)


class QueryUpdate:
    def __init__(self, base_table="", ):
        self.base_table = base_table
        self.table_renamings = collections.defaultdict(int)
        self.table_references = list()
        self.predicates = list()

    def include_subquery(self, subquery):

        # first up, build the rename map
        tables_in_sq = self._collect_tables(subquery)
        print("=== Tables found:", tables_in_sq)
        for table in tables_in_sq:
            self.table_renamings[table] += 1

    def _collect_tables(self, subquery):
        tables = []
        for clause in subquery["join"]["value"]["from"]:
            if "join" in clause:
                tables.append(clause["join"]["value"])
            else:
                tables.append(clause["value"])
        return tables

    def __str__(self) -> str:
        return f"Tables: {self.table_references}, Predicates: {self.predicates}"


def extract_subqueries(plan):
    from_clause = plan["from"]
    query_update = None

    # for each reference in the 'from' clause, check if it constitutes a subquery reference
    for table in from_clause:

        # extract the base table name and proceed with the joined tables
        if not isinstance(table, dict) or "join" not in table:
            query_update = QueryUpdate(table)
            continue

        join_target = table["join"]["value"]
        if isinstance(join_target, dict) and "select" in join_target:
            print("Found subquery:", join_target)
            print("#####")
            query_update.include_subquery(table)

    print(query_update)


q: Dict[Any, Any] = unwrap(df[df.label == "18a"].sql)
extract_subqueries(q)
