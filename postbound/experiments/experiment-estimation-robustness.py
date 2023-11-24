from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import datetime
from typing import Literal, Optional

import pandas as pd

from postbound import postbound as pb
from postbound.qal import base, qal
from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import jointree, presets
from postbound.optimizer.policies import cardinalities as cards
from postbound.util import jsonize, proc

OutDir = "robustness-shift"
BaselineFilling = 0.6
ShiftStep = 0.05
ShiftSpan = 0.4

job = workloads.job()

pg_instance = postgres.connect(config_file=".psycopg_connection_job")
pg_instance.cache_enabled = False
pg_stats = pg_instance.statistics()
pg_stats.cache_enabled = False
pg_stats.emulated = True
workload_shifter = postgres.WorkloadShifter(pg_instance)

fact_table = base.TableReference("title")
order_column = base.ColumnReference("production_year", fact_table)

ues_optimizer = pb.TwoStageOptimizationPipeline(pg_instance)
ues_optimizer = ues_optimizer.load_settings(presets.fetch("ues")).build()


def obtain_baseline_plans(outfile: str) -> None:
    tuple_drop_pct = 1 - BaselineFilling

    proc.run_cmd(["./workload-job-setup.sh", "--force"], work_dir=os.environ["PG_CTL_DIR"])
    workload_shifter.remove_ordered(order_column, row_pct=tuple_drop_pct, vacuum=True)

    native_plans: dict[str, jointree.PhysicalQueryPlan] = {}
    ues_plans: dict[str, jointree.PhysicalQueryPlan] = {}
    for label, query in job.entries():
        native_plan = pg_instance.optimizer().query_plan(query)
        native_plans[label] = jointree.PhysicalQueryPlan.load_from_query_plan(native_plan, query)
    for label, query in job.entries():
        # we obtain native and robust plans in two separate loops to ensure that the native plans are not influenced by any
        # settings that are set for the robust plans
        ues_plan = ues_optimizer.query_execution_plan(query)
        ues_plans[label] = ues_plan

    baseline = {"native_plans": native_plans, "robust_plans": ues_plans}
    with open(outfile, "w") as out:
        out.write(jsonize.to_json(baseline))


ExperimentType = Literal["native", "robust", "native-true-cards", "native-fixed"]


@dataclasses.dataclass
class DataShiftResult:
    fill_ratio: float
    plan_type: ExperimentType
    label: str
    query: str
    query_plan: str
    total_runtime: float


def obtain_data_shift_result(fill_ratio: float, label: str, query: qal.SqlQuery, plan_type: ExperimentType, *,
                             query_plans: Optional[dict] = None,
                             cardinality_estimator: Optional[cards.CardinalityHintsGenerator] = None) -> DataShiftResult:
    if plan_type == "native":
        explain_query = query
    elif plan_type == "robust":
        ues_plan = jointree.read_from_json(query_plans["robust_plans"][label])
        explain_query = pg_instance.hinting().generate_hints(query, ues_plan)
    elif plan_type == "native-fixed":
        native_plan = jointree.read_from_json(query_plans["native_plans"][label])
        explain_query = pg_instance.hinting().generate_hints(query, native_plan)
    elif plan_type == "native-true-cards":
        cardinality_hints = cardinality_estimator.estimate_cardinalities(query)
        explain_query = pg_instance.hinting().generate_hints(query, plan_parameters=cardinality_hints)
    else:
        raise ValueError("Unknown experiment type: {}".format(plan_type))

    start_time = datetime.now()
    explain_plan = pg_instance.optimizer().analyze_plan(explain_query)
    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()

    return DataShiftResult(fill_ratio=fill_ratio, plan_type=plan_type, label=label,
                           query=str(query), query_plan=jsonize.to_json(explain_plan),
                           total_runtime=total_runtime)


def simulate_data_shift(baseline_file: str, outfile: str) -> None:
    total_tuples = pg_instance.statistics().total_rows(fact_table)
    tuples_to_drop = ShiftStep * total_tuples
    cardinality_estimator = cards.PreciseCardinalityHintGenerator(pg_instance, enable_cache=True)
    with open(baseline_file, "r") as baselines:
        query_plans: dict = json.load(baselines)

    results: list[DataShiftResult] = []
    for data_step in range(BaselineFilling + ShiftSpan, BaselineFilling - ShiftSpan - ShiftStep, -ShiftStep):
        for label, query in job.entries():
            pg_instance.prewarm_tables(query.tables())
            results.append(obtain_data_shift_result(data_step, label, query, "native"))

            pg_instance.prewarm_tables(query.tables())
            results.append(obtain_data_shift_result(data_step, label, query, "native-fixed", query_plans))

            pg_instance.prewarm_tables(query.tables())
            results.append(obtain_data_shift_result(data_step, label, query, "native-true-cards",
                                                    cardinality_estimator=cardinality_estimator))

            pg_instance.prewarm_tables(query.tables())
            results.append(obtain_data_shift_result(data_step, label, query, "robust", query_plans))

        workload_shifter.remove_ordered(order_column, n_rows=tuples_to_drop, vacuum=True)
        cardinality_estimator.reset_cache()

    result_df = pd.DataFrame(results)
    result_df.to_csv(outfile)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "baseline" or mode == "full":
        os.makedirs(OutDir)
        obtain_baseline_plans(OutDir + "/baseline.json")
    if mode == "full":
        proc.run_cmd(["./workload-job-setup.sh", "--force"], work_dir=os.environ["PG_CTL_DIR"])
        pg_instance.reset_connection()
    if mode == "shift" or mode == "full":
        simulate_data_shift(OutDir + "/baseline.json", OutDir + "/data-shift.csv")


if __name__ == "__main__":
    main()
