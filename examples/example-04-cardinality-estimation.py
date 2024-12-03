#!/usr/bin/env python3
#
# This script shows how PostBOUND can be used to implement different cardinality estimation approaches. It implements a simple
# estimator that uses native estimates of an actual database system, but distorts the estimates by a random factor.
#
# Requirements: a running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities in the root postgres directory.
#


import random
import warnings
from typing import Optional

import postbound as pb
from postbound import db, qal, optimizer, util
from postbound.optimizer import jointree

warnings.simplefilter("ignore")


class JitteringCardinalityEstimator(pb.ParameterGeneration):
    # The entire estimation algorithm is implemented in this class. It satisfies the interface of the corresponding
    # optimization stage.

    def __init__(self, native_optimizer: db.OptimizerInterface) -> None:
        super().__init__()
        self.native_optimizer = native_optimizer

    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[optimizer.PhysicalOperatorAssignment]
                                 ) -> optimizer.PlanParameterization:
        # This is the most important method that handles the actual cardinality estimation

        # We store our cardinalities in this object
        cardinalities = optimizer.PlanParameterization()

        # Now, we need to iterate over all potential intermediate results of the query to generate an estimate for all of them
        # This is a drawback of the two-stage optimization approach used in PostBOUND: the actual physical database system has
        # no way to "call-back" to PostBOUND to request a new estimate because it does not know about PostBOUND's existence in
        # the first place. Therefore, we have to already pre-generate all information that could potentially become useful
        # within PostBOUND.
        # In our case, the easiest way to do so is to construct the powerset of all tables in the query. This spans all
        # possible intermediate results.

        for join in util.collections.powerset(query.tables()):
            if not join:
                # skip the empty set
                continue

            # In order to obtain the cardinality estimate of the native optimizer of the actual database system, we simulate
            # the query execution of our current intermediate result. This is done by constructing a specialized SQL query that
            # only handles the computation of the intermediate result. Afterwards, we can ask the optimizer for its cardinality
            # estimate of the specialized query to obtain the cardinality estimate for the intermediate result.
            query_fragment = qal.transform.extract_query_fragment(query, join)
            query_fragment = qal.transform.as_star_query(query_fragment)
            native_estimate = self.native_optimizer.cardinality_estimate(query_fragment)

            # Apply the distortion to the estimated cardinality
            estimate_devitation = random.random()
            estimated_cardinality = int(estimate_devitation * native_estimate)

            # And finally store the new cardinality estimate
            cardinalities.add_cardinality_hint(join, estimated_cardinality)

        return cardinalities

    def describe(self) -> dict:
        return {"name": "random_cardinalities"}


# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = pb.db.postgres.connect()
job_workload = pb.workloads.job()

# Now let's generate the optimization pipeline with our new cardinality estimator
pipeline = pb.TwoStageOptimizationPipeline(postgres_db)
pipeline.setup_plan_parameterization(JitteringCardinalityEstimator(postgres_db.optimizer()))
pipeline.build()

# Run a couple of optimizations. Hopefully, they each contain a different query plan with different cardinality estimates
query = job_workload["1a"]
for i in range(3):
    print("Run", i+1, "::")
    optimized_query = pipeline.optimize_query(query)
    query_plan = postgres_db.optimizer().query_plan(optimized_query)
    print(query_plan.inspect(skip_intermediates=True))
    print("--- --- ---")

# If you check the output carefully, you notice a caveat: The estimates of the base tables do not change. This is because the
# tool we use for cardinality injection with Postgres only works for intermediate results/joins. Therefore, even though our
# estimation algorithm produced different estimates for the base tables, these estimates had to be ignored in the query.
# If you take a look at the hint block, you will notice that cardinality hints for base tables indeed do not exist. Disabling
# the warnings-filter in line 22 will also show warnings during the query generation.
