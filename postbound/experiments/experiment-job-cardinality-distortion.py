
from __future__ import annotations

import dataclasses
import json
import math
import typing
from datetime import datetime

import pandas as pd

from postbound.db import db, postgres
from postbound.qal import base, qal, parser, transform
from postbound.experiments import workloads
from postbound.optimizer import jointree, planparams
from postbound.optimizer.policies import cardinalities
from postbound.util import jsonize, logging, dicts as dict_ext

CardAssignment = planparams.PlanParameterization
DistortionStep = 0.05
DefaultTimeout = 60 * 45
log = logging.make_logger(prefix=lambda: f"{logging.timestamp()} ::")

json_parser = parser.JsonParser()
pg_instance = postgres.connect(config_file=".psycopg_connection_job")
timeout_executor = postgres.TimeoutQueryExecutor(pg_instance)
job = workloads.job()


def _parse_tables(tabs: str) -> set[base.TableReference]:
    return {json_parser.load_table(t) for t in json.loads(tabs)}


class PreComputedCardinalities(cardinalities.CardinalityHintsGenerator):
    def __init__(self, workload: workloads.Workload, lookup_table_path: str) -> None:
        super().__init__(False)
        self._workload = workload
        self._true_card_df = pd.read_csv(lookup_table_path, converters={"tables": _parse_tables})

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[base.TableReference]) -> int:
        label = self._workload.label_of(query)
        relevant_samples = self._true_card_df[self._true_card_df["label"] == label]
        cardinality_sample = relevant_samples[relevant_samples["tables"] == tables]
        if len(cardinality_sample) == 0:
            raise ValueError("No matching sample found")
        elif len(cardinality_sample) > 1:
            raise ValueError("More than one matching sample found")
        cardinality = cardinality_sample.iloc[0]["cardinality"]
        return int(cardinality)

    def describe(self) -> dict:
        return {"name": "pre-computed cardinalities"}


card_generator = PreComputedCardinalities(job, "results/job/job-intermediate-cardinalities.csv")


def obtain_query_plan(query: qal.SqlQuery,
                      cardinality_hints: planparams.PlanParameterization) -> tuple[jointree.PhysicalQueryPlan, qal.SqlQuery]:
    hinted_query = pg_instance.hinting().generate_hints(query, plan_parameters=cardinality_hints)
    explain_plan = pg_instance.optimizer().query_plan(hinted_query)
    return jointree.PhysicalQueryPlan.load_from_query_plan(explain_plan, query), hinted_query


@dataclasses.dataclass
class DistortionResult:
    label: str
    query: str
    distortion_factor: float
    distortion_type: typing.Literal["overestimation", "underestimation"]
    query_plan: dict
    runtime: float
    cached: bool
    query_hints: str = ""


observed_plans: dict[jointree.PhysicalQueryPlan, tuple[float, db.QueryExecutionPlan]] = dict_ext.CustomHashDict(
    jointree.PhysicalQueryPlan.plan_hash)

distortion_results: list[DistortionResult] = []
for current_distortion_step in range(1, 20):
    underest_distortion = max(round(1 - current_distortion_step * DistortionStep, 2), 0)
    log("Simulating underestimation workload for pct", underest_distortion)
    card_distortion = cardinalities.CardinalityDistortion(card_generator, underest_distortion, distortion_strategy="fixed")

    for label, query in job.entries():
        log("Executing query", label, "for underestimation pct", underest_distortion)
        pg_instance.prewarm_tables(query.tables())
        underest_assignment = card_distortion.estimate_cardinalities(query)

        underest_plan, card_query = obtain_query_plan(query, underest_assignment)
        if underest_plan in observed_plans:
            log("Reusing existing plan for query", label, "at underestimation pct", underest_distortion)
            underest_runtime, underest_explain = observed_plans[underest_plan]
            cached = True
        else:
            try:
                start_time = datetime.now()
                underest_query = pg_instance.hinting().generate_hints(query, underest_plan)
                underest_query = transform.as_explain_analyze(underest_query)
                underest_explain = timeout_executor.execute_query(underest_query, DefaultTimeout)
                end_time = datetime.now()
                underest_runtime = (end_time - start_time).total_seconds()
            except TimeoutError:
                underest_runtime = math.inf
                underest_explain = []
            observed_plans[underest_plan] = underest_runtime, underest_explain
            cached = False

        distortion_results.append(DistortionResult(label, query,
                                                   underest_distortion, "underestimation",
                                                   jsonize.to_json(underest_explain), underest_runtime,
                                                   cached=cached, query_hints=card_query.hints))

for current_distortion_step in range(1, 40 + 1):
    overest_distortion = round(1 + current_distortion_step * DistortionStep, 2)
    log("Simulating overestimation workload for pct", overest_distortion)
    card_distortion = cardinalities.CardinalityDistortion(card_generator, overest_distortion, distortion_strategy="fixed")

    for label, query in job.entries():
        log("Executing query", label, "for overestimation pct", overest_distortion)

        pg_instance.prewarm_tables(query.tables())

        overest_assignment = card_distortion.estimate_cardinalities(query)
        overest_plan, card_query = obtain_query_plan(query, overest_assignment)
        if overest_plan in observed_plans:
            log("Reusing existing plan for query", label, "at overestimation pct", overest_distortion)
            overest_runtime, overest_explain = observed_plans[overest_plan]
            cached = True
        else:
            try:
                start_time = datetime.now()
                overest_query = pg_instance.hinting().generate_hints(query, overest_plan)
                overest_query = transform.as_explain_analyze(overest_query)
                overest_explain = timeout_executor.execute_query(overest_query, DefaultTimeout)
                end_time = datetime.now()
                overest_runtime = (end_time - start_time).total_seconds()
            except TimeoutError:
                overest_runtime = math.inf
                overest_explain = []
            observed_plans[overest_plan] = overest_runtime, overest_explain
            cached = False

        distortion_results.append(DistortionResult(label, query,
                                                   overest_distortion, "overestimation",
                                                   jsonize.to_json(overest_explain), overest_runtime,
                                                   cached=cached, query_hints=card_query.hints))

results_df = pd.DataFrame(distortion_results)
results_df.to_csv("results/job-pg-cardinality-distortion.csv", index=False)
