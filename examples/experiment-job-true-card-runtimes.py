from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from postbound import qal
from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import planparams
from postbound.util import jsonize


@dataclass
class BenchmarkResult:
    label: str
    query: str
    hints: str
    execution_time: float
    query_plan: str
    db_config: str


def parse_tables_list(tables: str) -> set[qal.TableReference]:
    jsonized = json.loads(tables)
    return {qal.TableReference(tab["full_name"], tab.get("alias")) for tab in jsonized}


def true_cardinalities(label: str) -> planparams.PlanParameterization:
    relevant_queries = card_df[card_df["label"] == label]
    plan_params = planparams.PlanParameterization()
    for __, row in relevant_queries.iterrows():
        plan_params.add_cardinality_hint(row["tables"], row["cardinality"])
    return plan_params


workloads.workloads_base_dir = "workloads/"
pg_db = postgres.connect(config_file=".psycopg_connection_job")
pg_db.cache_enabled = False
db_config = pg_db.describe()

print("Reading true cardinalities")
card_df = pd.read_csv("results/job/job-intermediate-cardinalities.csv",
                      converters={"tables": parse_tables_list})

benchmark_results = []
print("Starting workload execution")
for label, query in workloads.job().entries():
    print("Now executing query", query)
    pg_db.prewarm_tables(query.tables())
    original_query = query
    query = pg_db.hinting().generate_hints(query, None, None, true_cardinalities(label))
    query_start = datetime.now()
    pg_db.execute_query(query)
    query_end = datetime.now()
    execution_time = (query_end - query_start).total_seconds()
    query_plan = pg_db.execute_query(qal.transform.as_explain_analyze(query))
    result_wrapper = BenchmarkResult(label, str(original_query), query.hints.query_hints,
                                     execution_time, jsonize.to_json(query_plan),
                                     jsonize.to_json(db_config))
    benchmark_results.append(result_wrapper)

df = pd.DataFrame(benchmark_results)
os.makedirs("results/job/", exist_ok=True)
outfile = sys.argv[1] if len(sys.argv) > 1 else "job-true-card-runtimes.csv"
df.to_csv(f"results/job/{outfile}", index=False)
