
from __future__ import annotations

import os

from postbound.db import postgres
from postbound.experiments import runner, workloads, analysis
from postbound.util import logging

NumAnalyzeRuns = 10

logger = logging.make_logger(enabled=True, prefix=logging.timestamp)
pg_imdb = postgres.connect(config_file=".psycopg_connection_job")
job_workload = workloads.job()
query_config = runner.QueryPreparationService(analyze=True, prewarm=True, preparatory_statements=["SET geqo = 'off';"])


def update_stats(repetition: int) -> None:
    pg_imdb.statistics().update_statistics(perfect_mcv=True)
    os.system(f"pg_dump --table=pg_statistic --format=t --file=results/local/stats_dump_{repetition}.tar imdb")


update_stats("base")
result_df = runner.execute_workload(job_workload, pg_imdb, workload_repetitions=NumAnalyzeRuns, query_preparation=query_config,
                                    post_repetition_callback=update_stats, logger=logger)
results_df = analysis.prepare_export(result_df)
result_df.to_csv("results/local/pg-analyze-stability.csv", index=False)
