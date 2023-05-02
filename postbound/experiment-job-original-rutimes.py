from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from postbound.db import postgres
from postbound.experiments import workloads


@dataclass
class BenchmarkResult:
    label: str
    query: str
    execution_time: float


workloads.workloads_base_dir = "../workloads/"
pg_db = postgres.connect(config_file=".psycopg_connection_job")

benchmark_results = []
for label, query in workloads.job().entries():
    pg_db.prewarm_tables(query.tables())
    query_start = datetime.now()
    pg_db.execute_query(query, cache_enabled=False)
    query_end = datetime.now()
    execution_time = (query_end - query_start).total_seconds()
    result_wrapper = BenchmarkResult(label, str(query), execution_time)
    benchmark_results.append(result_wrapper)

df = pd.DataFrame(benchmark_results)
os.makedirs("results/ax1/")
df.to_csv("results/ax1/job-native-runtimes.csv", index=False)
