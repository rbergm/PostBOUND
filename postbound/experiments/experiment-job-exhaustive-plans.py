
from __future__ import annotations

import argparse
import dataclasses
import math
from datetime import datetime

import pandas as pd

from postbound.db import postgres
from postbound.experiments import workloads, analysis
from postbound.qal import qal, transform
from postbound.optimizer import jointree
from postbound.optimizer.strategies import enumeration
from postbound.util import logging


@dataclasses.dataclass
class ExperimentResult:
    label: str
    query: qal.SqlQuery
    execution_time: float
    timeout: bool
    query_result: object
    db_config: object


def load_native_plan(query: qal.SqlQuery, plan: jointree.PhysicalQueryPlan, database: postgres.PostgresInterface) -> object:
    hinted_query = database.hinting().generate_hints(query, plan)
    explain_query = transform.as_explain(hinted_query)
    return database.execute_query(explain_query, cache_enabled=False)


def benchmark_query(label: str, query: qal.SqlQuery, *,
                    database: postgres.PostgresInterface, timeout: float = 60.0,
                    max_n_plans: int, logger: logging.Logger) -> list[ExperimentResult]:
    exhaustive_enumerator = enumeration.ExhaustivePlanEnumerator(operator_args=dict(database=database))
    plans_to_evaluate: list[jointree.PhysicalQueryPlan] = []
    for i, plan in enumerate(exhaustive_enumerator.all_plans_for(query)):
        if i >= max_n_plans:
            # We cannot generate all plans for the query exhaustively. Therefore, no evaluation is possible.
            logger("Skipping query", label, "because it cannot be evaluated exhaustively.")
            return []
        plans_to_evaluate.append(plan)

    db_config = database.describe()
    timeout_executor = postgres.TimeoutQueryExecutor(database)
    n_timeouts = 0
    results: list[ExperimentResult] = []
    logger("Evaluating", len(plans_to_evaluate), "plans for query", label)
    for plan in plans_to_evaluate:
        try:
            start_time = datetime.now()
            result = timeout_executor.execute_query(query, plan, timeout)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            results.append(ExperimentResult(label=label, query=query, execution_time=execution_time, timeout=False,
                                            query_result=result, db_config=db_config))
        except TimeoutError:
            n_timeouts += 1
            explain_plan = load_native_plan(query, plan, database)
            execution_time = math.nan
            results.append(ExperimentResult(label=label, query=query, execution_time=execution_time, timeout=True,
                                            query_result=explain_plan, db_config=db_config))

    logger("Evaluated query", label, "(timeouts: ", n_timeouts, "total)")
    return results


def evaluate_workload(*, outfile: str, max_n_plans: int, timeout: float, verbose: bool) -> None:
    pg_imdb = postgres.connect(config_file=".psycopg_connection_job")
    job_workload = workloads.job()
    logger = logging.make_logger(enabled=verbose, prefix=logging.timestamp)

    results: list[ExperimentResult] = []
    for label, query in job_workload.entries():
        logger("Now evaluating query", label)
        results.append(benchmark_query(label, query, database=pg_imdb, timeout=timeout, max_n_plans=max_n_plans,
                                       logger=logger))

    result_df = pd.DataFrame(results)
    result_df = analysis.prepare_export(result_df)
    result_df.to_csv(outfile, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile",  "-o", type=str, required=True, help="The file to write the results to")
    parser.add_argument("--max-plans", "-n", type=int, default=1000, help="The maximum number of plans to evaluate per query")
    parser.add_argument("--timeout",  "-t", type=float, default=30.0, help="The timeout for each query evaluation")
    parser.add_argument("--verbose",  "-v", action="store_true", help="Whether to print debug information")

    args = parser.parse_args()
    evaluate_workload(outfile=args.outfile, max_n_plans=args.max_plans, timeout=args.timeout, verbose=args.verbose)


if __name__ == "main":
    main()
