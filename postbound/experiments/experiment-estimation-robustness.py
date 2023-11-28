from __future__ import annotations

import dataclasses
import json
import pathlib
import os
import sys
import textwrap
from datetime import datetime
from typing import Literal, Optional

import pandas as pd

from postbound import postbound as pb
from postbound.qal import base, qal
from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import jointree, presets
from postbound.optimizer.policies import cardinalities as cards
from postbound.util import jsonize, logging, proc

OutDir = "robustness-shift"
BaselineFilling = 0.6
ShiftStep = 0.05
ShiftSpan = 0.4
log = logging.make_logger(prefix=lambda: f"{logging.timestamp} ::")
delete_marker_file = pathlib.Path(OutDir, "delete-markers.csv").absolute()

job = workloads.job()

pg_instance = postgres.connect(config_file=".psycopg_connection_job", cache_enabled=False)
pg_instance.cache_enabled = False
pg_stats = pg_instance.statistics()
pg_stats.cache_enabled = False
pg_stats.emulated = True
workload_shifter = postgres.WorkloadShifter(pg_instance)
pg_config = pg_instance.current_configuration()

fact_table = base.TableReference("title")
fact_marker_column = base.ColumnReference("id", fact_table)
marker_table = base.TableReference(f"{fact_table.full_name}_delete_marker")
marker_column = base.ColumnReference(f"{fact_table.full_name}_{fact_marker_column.name}", marker_table)

ues_optimizer = pb.TwoStageOptimizationPipeline(pg_instance)
ues_optimizer = ues_optimizer.load_settings(presets.fetch("ues")).build()


def obtain_baseline_plans(outfile: str) -> None:
    workload_shifter.generate_marker_table(fact_table.full_name, 1 - BaselineFilling + ShiftSpan)
    workload_shifter.export_marker_table(fact_table.full_name, out_file=delete_marker_file)
    cursor = pg_instance.cursor()
    cursor.execute(f"CREATE TEMP TABLE delete_marker_buffer LIKE {marker_table.full_name}")

    total_marked_tuples = pg_instance.statistics().total_rows(marker_table)
    max_marker_idx = round(BaselineFilling * total_marked_tuples)
    baseline_marker_query = textwrap.dedent(f"""
                                            INSERT INTO delete_marker_buffer (marker_idx, {marker_column.name})
                                            SELECT * FROM {marker_table.full_name}
                                            WHERE marker_idx <= {max_marker_idx}""")
    cursor.execute(baseline_marker_query)
    workload_shifter.remove_marked(fact_table.full_name, marker_table="delete_marker_buffer", vacuum=True)

    native_plans: dict[str, jointree.PhysicalQueryPlan] = {}
    ues_plans: dict[str, jointree.PhysicalQueryPlan] = {}
    for label, query in job.entries():
        log("Obtaining native plan for query", label)
        native_plan = pg_instance.optimizer().query_plan(query)
        native_plans[label] = jointree.PhysicalQueryPlan.load_from_query_plan(native_plan, query)

    pg_instance.statistics().cache_enabled = True
    for label, query in job.entries():
        # we obtain native and robust plans in two separate loops to ensure that the native plans are not influenced by any
        # settings that are set for the robust plans
        log("Obtaining UES plan for query", label)
        ues_plan = ues_optimizer.query_execution_plan(query)
        ues_plans[label] = ues_plan
    pg_instance.statistics().cache_enabled = False
    pg_instance.reset_cache()

    baseline = {"native_plans": native_plans, "robust_plans": ues_plans}
    with open(outfile, "w") as out:
        out.write(jsonize.to_json(baseline))


ExperimentType = Literal["native", "robust", "native-true-cards", "native-fixed", "robust-fixed"]


@dataclasses.dataclass
class DataShiftResult:
    fill_ratio: float
    plan_type: ExperimentType
    label: str
    query: str
    query_plan: str
    total_runtime: float
    db_config: str


def obtain_data_shift_result(fill_ratio: float, label: str, query: qal.SqlQuery, plan_type: ExperimentType, *,
                             query_plans: Optional[dict] = None,
                             cardinality_estimator: Optional[cards.CardinalityHintsGenerator] = None) -> DataShiftResult:
    log("Executing", plan_type, "query", label, "at fill factor", fill_ratio)
    if plan_type == "native":
        explain_query = query
    elif plan_type == "robust":
        explain_query = ues_optimizer.optimize_query(query)
    elif plan_type == "robust-fixed":
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

    pg_instance.apply_configuration(pg_config)

    return DataShiftResult(fill_ratio=fill_ratio, plan_type=plan_type, label=label,
                           query=str(query), query_plan=jsonize.to_json(explain_plan),
                           total_runtime=total_runtime, db_config=jsonize.to_json(pg_instance.describe()))


def simulate_data_shift(baseline_file: str, outfile: str) -> None:
    total_n_tuples = pg_instance.statistics().total_rows(fact_table)
    tuples_to_drop: int = round(ShiftStep * total_n_tuples)
    workload_shifter.import_marker_table(target_table=fact_table.full_name, in_file=delete_marker_file)
    pg_instance.cursor().execute(f"CREATE TEMP TABLE delete_marker_buffer LIKE {fact_table.full_name}_delete_marker")

    cardinality_estimator = cards.PreciseCardinalityHintGenerator(pg_instance, enable_cache=True)
    with open(baseline_file, "r") as baselines:
        query_plans: dict = json.load(baselines)

    results: list[DataShiftResult] = []
    start_marker_idx, end_marker_idx = 0, tuples_to_drop
    for data_step in range(BaselineFilling + ShiftSpan, BaselineFilling - ShiftSpan - ShiftStep, -ShiftStep):
        log("Now at data shift pct", data_step)

        pg_instance.statistics().cache_enabled = True
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

        cardinality_estimator.reset_cache()
        pg_instance.statistics().cache_enabled = False
        pg_instance.reset_cache()

        pg_instance.cursor().execute("DELETE FROM delete_marker_buffer")
        marker_inflation_query = textwrap.dedent(f"""
                                                 INSERT INTO delete_marker_buffer (marker_idx, {marker_column.name})
                                                 SELECT * FROM {marker_table.full_name}
                                                 WHERE marker_idx BETWEEN {start_marker_idx} AND {end_marker_idx}""")
        pg_instance.cursor().execute(marker_inflation_query)
        workload_shifter.remove_marked(fact_table.full_name, marker_table="delete_marker_buffer", vacuum=True)
        start_marker_idx = end_marker_idx
        end_marker_idx += tuples_to_drop

    result_df = pd.DataFrame(results)
    result_df.to_csv(outfile)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "baseline" or mode == "full":
        os.makedirs(OutDir, exist_ok=True)
        log("Setting up fresh IMDB instance")
        proc.run_cmd(["./workload-job-setup.sh", "--force"], work_dir=os.environ["PG_CTL_PATH"])
        pg_instance.reset_connection()
        log("Obtaining baseline plans")
        obtain_baseline_plans(OutDir + "/baseline.json")
    if mode == "full":
        log("Resetting IMDB instance")
        proc.run_cmd(["./workload-job-setup.sh", "--force"], work_dir=os.environ["PG_CTL_PATH"])
        pg_instance.reset_connection()
    if mode == "shift" or mode == "full":
        log("Simulating data shift")
        simulate_data_shift(OutDir + "/baseline.json", OutDir + "/data-shift.csv")


if __name__ == "__main__":
    main()
