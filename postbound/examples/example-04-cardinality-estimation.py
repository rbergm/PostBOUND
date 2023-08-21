#!/usr/bin/env python3

import random
import warnings
from typing import Optional

from postbound import postbound as pb
from postbound.qal import qal, transform
from postbound.db import db, postgres
from postbound.experiments import workloads
from postbound.optimizer import stages, jointree, physops, planparams
from postbound.util import collections as collection_utils

warnings.simplefilter("ignore")


class JitteringCardinalityEstimator(stages.ParameterGeneration):
    def __init__(self, optimizer: db.OptimizerInterface) -> None:
        super().__init__()
        self.optimizer = optimizer

    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> planparams.PlanParameterization:
        cardinalities = planparams.PlanParameterization()

        for join in collection_utils.powerset(query.tables()):
            if not join:
                # skip the empty set
                continue

            query_fragment = transform.extract_query_fragment(query, join)
            native_estimate = self.optimizer.cardinality_estimate(query_fragment)

            estimate_devitation = random.random()

            estimated_cardinality = int(estimate_devitation * native_estimate)
            cardinalities.add_cardinality_hint(join, estimated_cardinality)

        return cardinalities

    def describe(self) -> dict:
        return {"name": "random_cardinalities"}


postgres_db = postgres.connect()
job_workload = workloads.job()

pipeline = pb.TwoStageOptimizationPipeline(postgres_db)
pipeline.setup_plan_parameterization(JitteringCardinalityEstimator(postgres_db.optimizer()))
pipeline.build()

query = job_workload["1a"]
for i in range(3):
    print("Run", i+1, "::")
    optimized_query = pipeline.optimize_query(query)
    query_plan = postgres_db.optimizer().query_plan(optimized_query)
    print(query_plan.inspect(skip_intermediates=True))
    print("--- --- ---")
