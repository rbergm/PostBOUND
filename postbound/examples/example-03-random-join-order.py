#!/usr/bin/env python3

import random
import warnings
from typing import Optional

from postbound import postbound as pb
from postbound.qal import qal
from postbound.db import db, postgres
from postbound.experiments import workloads
from postbound.optimizer import stages, jointree, joingraph

warnings.simplefilter("ignore")


class RandomJoinOrderOptimizer(stages.JoinOrderOptimization):
    def __init__(self, database: Optional[db.Database] = None) -> None:
        super().__init__()
        self.database = database if database is not None else db.DatabasePool.current_database()

    def optimize_join_order(self, query: qal.SqlQuery) -> jointree.LogicalJoinTree:
        join_graph = joingraph.JoinGraph(query)
        join_tree = jointree.LogicalJoinTree()

        while join_graph.contains_free_tables():
            candidate_joins = join_graph.available_join_paths()
            candidate_tables = [path.target_table for path in candidate_joins]

            next_table = random.choice(candidate_tables)

            join_tree = join_tree.join_with_base_table(next_table)
            join_graph.mark_joined(next_table)

        return join_tree

    def describe(self) -> dict:
        return {"name": "random_linear"}


postgres_db = postgres.connect()
job_workload = workloads.job()

pipeline = pb.TwoStageOptimizationPipeline(postgres_db)
pipeline.setup_join_order_optimization(RandomJoinOrderOptimizer(postgres_db))
pipeline.build()

query = job_workload["1a"]
for i in range(3):
    print("Run", i+1, "::")
    optimized_query = pipeline.optimize_query(query)
    query_plan = postgres_db.optimizer().query_plan(optimized_query)
    print(query_plan.inspect(skip_intermediates=True))
    print("--- --- ---")
