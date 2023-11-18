from __future__ import annotations

import dataclasses
import json
import sys
from typing import Any

import pandas as pd

from postbound.db import postgres
from postbound.qal import qal, transform
from postbound.experiments import workloads


@dataclasses.dataclass
class PlannerResult:
    label: str
    query: qal.SqlQuery
    dynprog_plan: Any
    dynprog_optimization_time: float
    dynprog_execution_time: float
    geqo_plan: Any
    geqo_optimization_time: float
    geqo_execution_time: float


pg_instance = postgres.connect(config_file=".psycopg_connection_job")
db_cursor = pg_instance.cursor()
job = workloads.job()

planner_results: list[PlannerResult] = []
for label, query in job.entries():
    print(label, file=sys.stderr)
    pg_instance.prewarm_tables(query.tables())
    explain_query = transform.as_explain_analyze(query)

    db_cursor.execute("SET geqo = 'off';")
    dynprog_plan = postgres.PostgresExplainPlan(pg_instance.execute_query(explain_query, cache_enabled=False))

    db_cursor.execute("SET geqo = 'on'; SET geqo_threshold = 2;")
    geqo_plan = postgres.PostgresExplainPlan(pg_instance.execute_query(explain_query, cache_enabled=False))

    planner_result = PlannerResult(label, query,
                                   json.dumps(dynprog_plan.explain_data),
                                   dynprog_plan.planning_time, dynprog_plan.execution_time,
                                   json.dumps(geqo_plan.explain_data), geqo_plan.planning_time, geqo_plan.execution_time)
    planner_results.append(planner_result)


result_df = pd.DataFrame(planner_results)
result_df.to_csv("postgres-optimization-times.csv", index=False)
