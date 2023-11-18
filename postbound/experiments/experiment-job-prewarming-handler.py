from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd

from postbound.db import postgres
from postbound.qal import qal, transform
from postbound.experiments import workloads


@dataclasses.dataclass
class PrewarmingResult:
    label: str
    query: qal.SqlQuery
    cold_plan: Any
    cold_exec_time: float
    hot_plan: Any
    hot_exec_time: float


pg_instance = postgres.connect(config_file=".psycopg_connection_job")
job = workloads.job()
if len(sys.argv) > 1:
    allowed_labels = set(sys.argv[1:])
    print("Only executing for labels", allowed_labels)
    job = job.filter_by(lambda label, __: label in allowed_labels)

prewarming_results: list[PrewarmingResult] = []
for label, query in job.entries():
    explain_query = transform.as_explain_analyze(query)
    explain_query = pg_instance.hinting().format_query(explain_query)

    start_time = datetime.now()
    cold_plan = pg_instance.execute_query(explain_query, cache_enabled=False)
    end_time = datetime.now()
    cold_time = (end_time - start_time).total_seconds()

    pg_instance.prewarm_tables(query.tables())
    start_time = datetime.now()
    hot_plan = pg_instance.execute_query(explain_query, cache_enabled=False)
    end_time = datetime.now()
    hot_time = (end_time - start_time).total_seconds()

    prewarm_result = PrewarmingResult(label, query,
                                      json.dumps(cold_plan), cold_time,
                                      json.dumps(hot_plan), hot_time)
    prewarming_results.append(prewarm_result)

current_result_df = pd.DataFrame(prewarming_results)

result_file = "postgres-job-prewarming-influence.csv"
if os.path.exists(result_file):
    existing_results_df = pd.read_csv(result_file)
    result_df = pd.concat([existing_results_df, current_result_df])
else:
    result_df = current_result_df
result_df.to_csv(result_file, index=False)
