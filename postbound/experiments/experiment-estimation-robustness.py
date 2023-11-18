from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import datetime

import pandas as pd

from postbound import postbound as pb
from postbound.qal import base
from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import jointree, presets
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


@dataclasses.dataclass
class DataShiftResult:
    fill_ratio: float
    label: str
    query: str
    native_plan: dict
    native_runtime: float
    robust_plan: dict
    robust_runtime: float


def simulate_data_shift(baseline_file: str, outfile: str) -> None:
    total_tuples = pg_instance.statistics().total_rows(fact_table)
    tuples_to_drop = ShiftStep * total_tuples
    with open(baseline_file, "r") as baselines:
        query_plans: dict = json.load(baselines)

    results: list[DataShiftResult] = []
    for data_step in range(BaselineFilling + ShiftSpan, BaselineFilling - ShiftSpan - ShiftStep, -ShiftStep):
        for label, query in job.entries():
            native_query = jointree.read_from_json(query_plans["native_plans"][label])
            start_time = datetime.now()
            native_plan = pg_instance.optimizer().analyze_plan(native_query)
            end_time = datetime.now()
            native_runtime = (end_time - start_time).total_seconds()

            ues_query = jointree.read_from_json(query_plans["robust_plans"][label])
            start_time = datetime.now()
            ues_plan = pg_instance.optimizer().analyze_plan(ues_query)
            end_time = datetime.now()
            robust_runtime = (end_time - start_time).total_seconds()

            result = DataShiftResult(
                fill_ratio=data_step,
                label=label,
                query=query,
                native_plan=jsonize.to_json(native_plan),
                native_runtime=native_runtime,
                robust_plan=jsonize.to_json(ues_plan),
                robust_runtime=robust_runtime,
            )
            results.append(result)

        workload_shifter.remove_ordered(order_column, n_rows=tuples_to_drop, vacuum=True)

    result_df = pd.DataFrame(results)
    result_df.to_csv(outfile)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "baseline" or mode == "full":
        obtain_baseline_plans(OutDir + "/baseline.json")
    if mode == "full":
        proc.run_cmd(["./workload-job-setup.sh", "--force"], work_dir=os.environ["PG_CTL_DIR"])
        pg_instance.reset_connection()
    if mode == "shift" or mode == "full":
        simulate_data_shift(OutDir + "/baseline.json", OutDir + "/data-shift.csv")


if __name__ == "__main__":
    main()
