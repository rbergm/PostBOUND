
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from postbound.db import postgres
from postbound.experiments import workloads
from postbound.qal import parser, qal, transform
from postbound.optimizer import planparams
from postbound.util import collections as collection_utils

pg_instance = postgres.connect(config_file=".psycopg_connection_job")
job = workloads.job()

card_df = pd.read_csv("results/job/job-intermediate-cardinalities.csv")
card_df["tables"] = card_df["tables"].apply(json.loads)
card_df["tables"] = card_df["tables"].apply(lambda tabs: [parser.JsonParser().load_table(tab_data) for tab_data in tabs])


def true_cardinalities(label: str) -> planparams.PlanParameterization:
    relevant_queries = card_df[card_df["label"] == label]
    plan_params = planparams.PlanParameterization()
    for __, row in relevant_queries.iterrows():
        plan_params.add_cardinality_hint(row["tables"], row["cardinality"])
    return plan_params


filter_predicates = collection_utils.flatten(query.predicates().filters() for query in job.queries())
filter_columns = collection_utils.set_union(filter_pred.columns() for filter_pred in filter_predicates)
pg_instance.statistics().update_statistics(filter_columns, perfect_mcv=True)


@dataclass
class ExecutionResult:
    label: str
    query: qal.SqlQuery
    cost_model: Literal["default", "in-memory"]
    query_plan: Any
    execution_time: float


execution_results: list[ExecutionResult] = []
for label, query in job.entries():
    print(label, ",default", sep="", file=sys.stderr)
    pg_instance.prewarm_tables(query.tables())

    explain_query = transform.as_explain_analyze(query)
    explain_query = pg_instance.hinting().generate_hints(explain_query, None, None, true_cardinalities(label))

    pg_instance.cursor().execute("SET geqo = 'off';")
    start_time = datetime.now()
    query_plan = pg_instance.execute_query(explain_query, cache_enabled=False)
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    execution_result = ExecutionResult(label, query, "default", json.dumps(query_plan), execution_time)
    execution_results.append(execution_result)

for label, query in job.entries():
    print(label, ",in-memory", sep="", file=sys.stderr)
    pg_instance.prewarm_tables(query.tables())

    explain_query = transform.as_explain_analyze(query)
    explain_query = pg_instance.hinting().generate_hints(explain_query, None, None, true_cardinalities(label))

    pg_instance.cursor().execute("SET geqo = 'off'; SET random_page_cost = 1.1;")
    start_time = datetime.now()
    query_plan = pg_instance.execute_query(explain_query, cache_enabled=False)
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    execution_result = ExecutionResult(label, query, "in-memory", json.dumps(query_plan), execution_time)
    execution_results.append(execution_result)

result_df = pd.DataFrame(execution_results)
result_df.to_csv("job-optimal-runtimes.csv", index=False)
