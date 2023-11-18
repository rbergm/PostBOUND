
from __future__ import annotations

import dataclasses
import json
import typing
from datetime import datetime

import pandas as pd

from postbound.db import db, postgres
from postbound.qal import base, qal, parser
from postbound.experiments import workloads
from postbound.optimizer import jointree, planparams
from postbound.util import jsonize

CardAssignment = planparams.PlanParameterization
DistortionStep = 0.05

json_parser = parser.JsonParser()
pg_instance = postgres.connect(config_file=".psycopg_connection_job")
job = workloads.job()
true_card_df = pd.read_csv("results/job/job-intermediate-cardinalities.csv",
                           converters={"tables": lambda tabs: {json_parser.load_table(t) for t in json.loads(tabs)}})


class CardEstDistortion:
    def __init__(self, distortion_step: float, distortion_amount: int) -> None:
        self.distortion_step = distortion_step
        self.distortion_amount = distortion_amount

    def determine_parameterizations(self, label: str) -> tuple[CardAssignment, CardAssignment]:
        relevant_cardinalities = true_card_df[true_card_df["label"] == label]
        overestimation_factor = 1 + self.distortion_step * self.distortion_amount
        underestimation_factor = max(1 - self.distortion_step * self.distortion_amount, 0)
        print(underestimation_factor, overestimation_factor)

        overestimation_assignment = planparams.PlanParameterization()
        underestimation_assignment = planparams.PlanParameterization()
        for _, row in relevant_cardinalities.iterrows():
            tables: set[base.TableReference] = row["tables"]
            cardinality: int = row["cardinality"]
            overestimation_assignment.add_cardinality_hint(tables, round(overestimation_factor * cardinality))
            underestimation_assignment.add_cardinality_hint(tables, round(underestimation_factor * cardinality))

        return underestimation_assignment, overestimation_assignment


def obtain_query_plan(query: qal.SqlQuery, cardinality_hints: planparams.PlanParameterization) -> jointree.PhysicalQueryPlan:
    hinted_query = pg_instance.hinting().generate_hints(query, plan_parameters=cardinality_hints)
    explain_plan = pg_instance.optimizer().query_plan(hinted_query)
    return jointree.PhysicalQueryPlan.load_from_query_plan(explain_plan, query)


@dataclasses.dataclass
class DistortionResult:
    label: str
    query: str
    distortion_factor: float
    distortion_type: typing.Literal["overestimation", "underestimation"]
    query_plan: dict
    runtime: float


observed_plans: dict[jointree.PhysicalQueryPlan, tuple[float, db.QueryExecutionPlan]] = {}

distortion_results: list[DistortionResult] = []
for current_distortion in range(20):
    card_distortion = CardEstDistortion(DistortionStep, current_distortion)

    for label, query in job.entries():
        underest_assignment, overest_assignment = card_distortion.determine_parameterizations(label)
        underest_plan = obtain_query_plan(query, underest_assignment)
        if underest_plan in observed_plans:
            underest_runtime, underest_explain = observed_plans[underest_plan]
        else:
            start_time = datetime.now()
            underest_explain = pg_instance.optimizer().analyze_plan(underest_plan)
            end_time = datetime.now()
            underest_runtime = (end_time - start_time).total_seconds()
            observed_plans[underest_plan] = underest_runtime, underest_explain
        distortion_results.append(DistortionResult(label, query,
                                                   current_distortion, "underestimation",
                                                   jsonize.to_json(underest_explain), underest_runtime))

        overest_plan = obtain_query_plan(query, overest_assignment)
        if overest_plan in observed_plans:
            overest_runtime, overest_explain = observed_plans[overest_plan]
        else:
            start_time = datetime.now()
            overest_explain = pg_instance.optimizer().analyze_plan(overest_plan)
            end_time = datetime.now()
            overest_runtime = (end_time - start_time).total_seconds()
            observed_plans[overest_plan] = overest_runtime, overest_explain
        distortion_results.append(DistortionResult(label, query,
                                                   current_distortion, "overestimation",
                                                   jsonize.to_json(overest_explain), overest_runtime))

results_df = pd.DataFrame(distortion_results)
