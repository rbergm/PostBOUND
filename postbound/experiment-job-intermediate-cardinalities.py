#!/usr/bin/env python3

import collections
import warnings

import pandas as pd

from postbound.qal import qal, transform
from postbound.db import postgres
from postbound.experiments import workloads, analysis
from postbound.util import collections as collection_utils, jsonize

workloads.workloads_base_dir = "../workloads"

postgres_db = postgres.connect(config_file=".psycopg_connection_job")
db_pool = postgres.ParallelQueryExecutor(postgres_db.connect_string, n_threads=12)
job_benchmark = workloads.job(simplified=False)

explored_queries: set[qal.SqlQuery] = set()
fragment_to_queries_map: dict[qal.SqlQuery, list[qal.SqlQuery]] = collections.defaultdict(list)

for label, query in job_benchmark.entries():
    if not isinstance(query, qal.ImplicitSqlQuery):
        warnings.warn(f"Skipping query {label} b/c query fragments can currently "
                      "only be created for implicit queries.")
        continue

    tables = query.tables()
    query_predicates = query.predicates()
    candidate_joins = collection_utils.powerset(tables)

    for joined_tables in candidate_joins:
        if not joined_tables:
            continue
        if not query_predicates.joins_tables(joined_tables):
            continue

        query_fragment = transform.extract_query_fragment(query, joined_tables)
        assert query_fragment is not None
        query_fragment = transform.as_count_star_query(query_fragment)
        if query_fragment in explored_queries:
            continue
        explored_queries.add(query_fragment)
        fragment_to_queries_map[query_fragment].append(query)
        db_pool.queue_query(query_fragment)

print(".. All queries submitted to DB pool")
db_pool.drain_queue()
fragment_cardinalities: dict[qal.SqlQuery, int] = db_pool.result_set()

queries, fragments, labels, fragment_tables, cardinalities = [], [], [], [], []
for result_fragment, cardinality in fragment_cardinalities.items():
    for query in fragment_to_queries_map[result_fragment]:
        query_label = job_benchmark.label_of(query)
        queries.append(query)
        fragments.append(result_fragment)
        labels.append(query_label)
        fragment_tables.append(jsonize.to_json(list(result_fragment.tables())))
        cardinalities.append(cardinality)

result_df = pd.DataFrame({"label": labels, "query": queries,
                          "query_fragment": fragments, "tables": fragment_tables,
                          "cardinality": cardinalities})
result_df = analysis.sort_results(result_df, "label")
result_df.to_csv("job-intermediate-cardinalities.csv", index=False)
