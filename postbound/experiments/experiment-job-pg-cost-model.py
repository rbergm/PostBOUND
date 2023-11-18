
import dataclasses
import datetime
import json
import sys
import typing

import numpy as np
import pandas as pd

from postbound.db import postgres
from postbound.experiments import workloads
from postbound.qal import qal, transform


@dataclasses.dataclass
class CostModelResult:
    label: str
    query: qal.SqlQuery
    random_cost: float
    query_plan: typing.Any
    execution_time: float
    db_config: typing.Any


pg_instance = postgres.connect(config_file=".psycopg_connection_job")
job = workloads.job()
seq_cost = float(pg_instance.execute_query("SHOW seq_page_cost;", cache_enabled=False))
default_random_cost = float(pg_instance.execute_query("SHOW random_page_cost;", cache_enabled=False))

cost_model_results: list[CostModelResult] = []
for current_cost in np.arange(seq_cost, default_random_cost, 0.1):
    pg_instance.execute_query(f"SET random_page_cost = {current_cost};", cache_enabled=False)
    for label, query in job.entries():
        print(current_cost, label, sep=",", file=sys.stderr)
        pg_instance.prewarm_tables(query.tables())
        explain_query = transform.as_explain_analyze(query)

        start_time = datetime.datetime.now()
        query_plan = pg_instance.execute_query(explain_query, cache_enabled=False)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        cost_model_results.append(CostModelResult(label, query, current_cost,
                                                  json.dumps(query_plan), execution_time, json.dumps(pg_instance.describe())))

results_df = pd.DataFrame(cost_model_results)
results_df.to_csv("job-pg-cost-model.csv", index=False)
