
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import typing
from collections.abc import Iterable
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from postbound.db import db, postgres
from postbound.qal import base, qal, parser, transform
from postbound.experiments import workloads
from postbound.optimizer import jointree, planparams
from postbound.optimizer.policies import cardinalities
from postbound.util import jsonize, logging, dicts as dict_ext

DistortionType = typing.Literal["overestimation", "underestimation", "none"]
CardAssignment = planparams.PlanParameterization
ObservedPlansDict = dict[jointree.PhysicalQueryPlan, tuple[float, db.QueryExecutionPlan]]
DistortionStep = 0.05
DefaultTimeout = 60 * 45
DefaultOutfile = "results/job/job-pg-card-distortion.csv"
log = logging.make_logger(prefix=lambda: f"{logging.timestamp()} ::")


def _parse_tables(tabs: str) -> set[base.TableReference]:
    json_parser = parser.JsonParser()
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


def obtain_query_plan(query: qal.SqlQuery, cardinality_hints: planparams.PlanParameterization, *,
                      pg_instance: postgres.PostgresInterface) -> tuple[jointree.PhysicalQueryPlan, qal.SqlQuery]:
    hinted_query = pg_instance.hinting().generate_hints(query, plan_parameters=cardinality_hints)
    explain_plan = pg_instance.optimizer().query_plan(hinted_query)
    return jointree.PhysicalQueryPlan.load_from_query_plan(explain_plan, query), hinted_query


@dataclasses.dataclass
class DistortionResult:
    label: str
    query: str
    distortion_factor: float
    distortion_type: DistortionType
    query_plan: dict
    runtime: float
    cached: bool
    query_hints: str = ""


def write_distortion_results(results: Iterable[DistortionResult], out: str) -> None:
    fresh_results_df = pd.DataFrame(results)
    if not os.path.exists(out):
        fresh_results_df.to_csv(out, index=False)
        return
    existing_results_df = pd.read_csv(out)
    merged_df = pd.concat([existing_results_df, fresh_results_df])
    merged_df.to_csv(out, index=False)


def obtain_vanilla_results(workload: workloads.Workload[str], *, out: str, pg_instance: postgres.PostgresInterface,
                           card_generator: cardinalities.CardinalityHintsGenerator) -> ObservedPlansDict:
    distortion_results: list[DistortionResult] = []
    observed_plans = dict_ext.CustomHashDict(jointree.PhysicalQueryPlan.plan_hash)
    for label, query in workload.entries():
        log("Now obtaining vanilla results for query", label)
        pg_instance.prewarm_tables(query.tables())
        parameterization = card_generator.estimate_cardinalities(query)
        query_plan, hinted_query = obtain_query_plan(query, parameterization, pg_instance=pg_instance)
        hinted_query = transform.as_explain_analyze(hinted_query)

        start_time = datetime.now()
        explain_data = pg_instance.execute_query(hinted_query, cache_enabled=False)
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()

        observed_plans[query_plan] = total_runtime, explain_data
        distortion_results.append(DistortionResult(label, query, 1.0, "none", jsonize.to_json(explain_data), total_runtime,
                                                   cached=False, query_hints=hinted_query.hints))
    write_distortion_results(distortion_results, out)
    return observed_plans


def obtain_distortion_results(workload: workloads.Workload[str], distortion_factors: Iterable[float], *,
                              out: str, pg_instance: postgres.PostgresInterface,
                              card_generator: cardinalities.CardinalityHintsGenerator,
                              observed_plans: Optional[ObservedPlansDict] = None) -> ObservedPlansDict:
    timeout_executor = postgres.TimeoutQueryExecutor(pg_instance)
    observed_plans = (dict_ext.CustomHashDict(jointree.PhysicalQueryPlan.plan_hash) if observed_plans is None
                      else observed_plans)
    for current_distortion in distortion_factors:
        log("Simulating distorted workload for pct", current_distortion)

        distortion_results: list[DistortionResult] = []
        if current_distortion == 0:
            distortion_type = "none"
        elif current_distortion < 1.0:
            distortion_type = "underestimation"
        else:
            distortion_type = "overestimation"
        card_distortion = cardinalities.CardinalityDistortion(card_generator, current_distortion, distortion_strategy="fixed")

        for label, query in workload.entries():
            log("Executing query", label, "for distortion factor", current_distortion)
            pg_instance.prewarm_tables(query.tables())
            distorted_parameterization = card_distortion.estimate_cardinalities(query)

            query_plan, card_query = obtain_query_plan(query, distorted_parameterization, pg_instance=pg_instance)
            if query_plan in observed_plans:
                log("Reusing existing plan for query", label, "for distortion factor", current_distortion)
                total_runtime, explain_data = observed_plans[query_plan]
                cached = True
            else:
                # we better base our hinted query on the entire query plan to prevent Postgres from switching plans for the
                # cardinality hints
                hinted_query = pg_instance.hinting().generate_hints(query, query_plan)
                hinted_query = transform.as_explain_analyze(hinted_query)
                try:
                    start_time = datetime.now()
                    explain_data = timeout_executor.execute_query(hinted_query, DefaultTimeout)
                    end_time = datetime.now()
                    total_runtime = (end_time - start_time).total_seconds()
                except TimeoutError:
                    total_runtime = math.inf
                    hinted_query = transform.as_explain(hinted_query)
                    explain_data = pg_instance.execute_query(hinted_query, cache_enabled=False)
                observed_plans[query_plan] = total_runtime, explain_data
                cached = False

            distortion_results.append(DistortionResult(label, query,
                                                       current_distortion, distortion_type,
                                                       jsonize.to_json(explain_data), total_runtime,
                                                       cached=cached, query_hints=card_query.hints))

        write_distortion_results(distortion_results, out)

    return observed_plans


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment studying the effect of cardinality misestimates on query plans.")
    parser.add_argument("--include-vanilla", action="store_true", help="Include baseline plans without misestimates")
    parser.add_argument("--include-default-underest", action="store_true",
                        help="Simulate underestimation in range [0.05, 0.95] in steps of 0.05")
    parser.add_argument("--include-default-overest", action="store_true",
                        help="Simulate overestimation in range [1.05, 3.0] in steps of 0.05")
    parser.add_argument("--include-extreme-overest", action="store_true", help="Simulate very large overestimations")
    parser.add_argument("--distortion-factor", action="store", type=float, nargs="+", default=[],
                        help="Simulate custom misestimate factors")
    parser.add_argument("--base-cards", action="store", choices=["native", "actual"], default="actual",
                        help="Change baseline cardinalities to use for the misestimates. Allowed values are native "
                        "(using the Postgres optimizer), or the true cardinalities. The latter is the default option.")
    parser.add_argument("--queries", action="store", type=str, nargs="+", default=[],
                        help="Simulate misestimates only for the given query labels")
    parser.add_argument("-o", "--out", action="store", default=DefaultOutfile,
                        help=f"Output file to write the results to, defaults to {DefaultOutfile}")

    args = parser.parse_args()

    workload = workloads.job()
    if args.queries:
        workload = workload & set(args.queries)

    distortion_factors = np.empty(0)
    if args.include_default_underest:
        default_underest_factors = [max(round(1.0 - i * DistortionStep, 2), 0) for i in range(1, 20)]
        distortion_factors = np.concatenate([distortion_factors, default_underest_factors])
    if args.include_default_overest:
        default_overest_factors = [round(1.0 + i * DistortionStep, 2) for i in range(1, 40 + 1)]
        distortion_factors = np.concatenate([distortion_factors, default_overest_factors])
    if args.include_extreme_overest:
        extreme_estimation_points = [10, 25, 50, 75]
        extreme_estimation_factors = [10**i for i in range(7)]
        extreme_distortion_factors = sorted(np.outer(extreme_estimation_points, extreme_estimation_factors).flatten())
        distortion_factors = np.concatenate([distortion_factors, extreme_distortion_factors])
    distortion_factors = np.concatenate([distortion_factors, args.distortion_factor])
    if distortion_factors.size == 0 and not args.include_vanilla:
        parser.error("At least one distortion factor is required, but none were given.")

    pg_instance = postgres.connect(config_file=".psycopg_connection_job")
    card_generator = (PreComputedCardinalities(workload, "results/job/job-intermediate-cardinalities.csv")
                      if args.base_cards == "actual" else cardinalities.NativeCardinalityHintGenerator(pg_instance))
    observed_plans = dict_ext.CustomHashDict(jointree.PhysicalQueryPlan.plan_hash)

    if args.include_vanilla:
        log("Obtaining vanilla query runtimes")
        observed_plans = obtain_vanilla_results(workload, out=args.out,
                                                pg_instance=pg_instance, card_generator=card_generator)
    if distortion_factors.size > 0:
        log("Obtaining distorted query runtimes for factors", distortion_factors.tolist())
        obtain_distortion_results(workload, distortion_factors, out=args.out, pg_instance=pg_instance,
                                  card_generator=card_generator, observed_plans=observed_plans)


if __name__ == "__main__":
    main()
