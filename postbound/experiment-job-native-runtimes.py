from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from postbound.db import postgres
from postbound.experiments import workloads
from postbound.qal import transform
from postbound.util import jsonize


@dataclass
class BenchmarkResult:
    label: str
    query: str
    execution_time: float
    query_plan: str
    db_config: str


workloads.workloads_base_dir = "../workloads/"
pg_db = postgres.connect(config_file=".psycopg_connection_job")
pg_db.cache_enabled = False
db_config = pg_db.inspect()

benchmark_results = []
for label, query in workloads.job().entries():
    pg_db.prewarm_tables(query.tables())
    query_start = datetime.now()
    pg_db.execute_query(query)
    query_end = datetime.now()
    execution_time = (query_end - query_start).total_seconds()
    query_plan = pg_db.execute_query(transform.as_explain_analyze(query))
    result_wrapper = BenchmarkResult(label, str(query), execution_time, jsonize.to_json(query_plan),
                                     jsonize.to_json(db_config))
    benchmark_results.append(result_wrapper)

df = pd.DataFrame(benchmark_results)
os.makedirs("results/ax1/", exist_ok=True)
df.to_csv("results/ax1/job-native-runtimes.csv", index=False)
