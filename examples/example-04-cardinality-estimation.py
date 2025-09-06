#!/usr/bin/env python3
#
# This script shows how PostBOUND can be used to implement different cardinality estimation approaches. It implements a simple
# estimator that uses native estimates of an actual database system, but distorts the estimates by a random factor.
#
# Requirements: a running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities from db-support/postgres.
#

import math
import random
import warnings
from collections.abc import Iterable
from typing import Optional

import postbound as pb

warnings.simplefilter("ignore")

# In PostBOUND, there are two optimization pipelines that can work with cardinality estimates:
# - the TextBookOptimizationPipeline, which models the traditional optimizer architecture with plan enumerator, cost model and
#   cardinality estimator.
# - the MultiStageOptimizationPipeline, which first generates a logical join order and afterwards selects the optimal physical
#   operators for the join order. This pipeline is designed to let the optimizer of the target database to "fill the gaps" and
#   only perform part of the optimization process. For example, the pipeline can only specify the logical join order, in which
#   case the native optimizer selects its own physical operators. In this case, the cardinality estimates are used to guide the
#   operator selection of the native optimizer. (As an extreme case, one can also omit join order and physical operators in the
#   pipeline. This is equivalent to just overwriting the cardinality estimates)
#
# Since both pipelines require different interfaces for cardinality estimation, we would need to choose which one we want to
# implement. However, cardinality estimation is often agnostic to the specific usage scenario (text book or multi-stage).
# Therefore, the cardinalities module provides a common interface for both pipelines, called CardinalityGenerator.
# In this example, we are going to use that interface to implement a cardinality estimator that works in both pipelines.


class JitteringCardinalityEstimator(pb.CardinalityGenerator):
    # The entire estimation algorithm is implemented in this class. It satisfies the interface of the corresponding
    # optimization stage.

    def __init__(self, native_optimizer: pb.db.OptimizerInterface) -> None:
        super().__init__(False)
        self.native_optimizer = native_optimizer

    def calculate_estimate(
        self,
        query: pb.SqlQuery,
        tables: pb.TableReference | Iterable[pb.TableReference],
    ) -> pb.Cardinality:
        # For our current intermediate, we simply ask the native optimizer for its cardinality estimate. Afterwards, we distort
        # the estimate by a random factor.
        # To retrieve the native cardinality estimate, we need to construct a specialized SQL query that only computes the
        # intermediate result, but applies all joins and filters from the original query.
        # This is stored in the query_fragment.

        tables = pb.util.enlist(tables)
        query_fragment = pb.qal.transform.extract_query_fragment(query, tables)
        if not query_fragment:
            return math.nan
        query_fragment = pb.qal.transform.as_star_query(query_fragment)
        native_estimate = self.native_optimizer.cardinality_estimate(query_fragment)

        distortion_factor = random.random()
        estimated_cardinality = round(distortion_factor * native_estimate)
        return estimated_cardinality

    def generate_plan_parameters(
        self,
        query: pb.SqlQuery,
        join_order: Optional[pb.LogicalJoinTree],
        operator_assignment: Optional[pb.PhysicalOperatorAssignment],
    ) -> pb.PlanParameterization:
        # This method is specific to the MultiStageOptimizationPipeline
        # We actually do not need to implement this method, the CardinalityHintsGenerator interface already provides a decent
        # default implementation, which essentially performs the same steps as we do in this method.
        # We just show how we could implement it, if we wanted to or needed to.

        cardinalities = pb.PlanParameterization()

        # If we do not have a join order, we need to iterate over all potential intermediate results of the query to generate
        # an estimate for all of them.
        # This is a drawback of the multi-stage optimization approach used in PostBOUND: the actual physical database system
        # has no way to "call-back" to PostBOUND to request a new estimate because it does not know about PostBOUND's existence
        # in the first place. Therefore, we have to pre-generate all information within PostBOUND that could potentially become
        # useful for the native optimizer of the physical database system.
        # In our case, the easiest way to do so is to construct the powerset of all tables in the query. This spans all
        # possible intermediate results.
        #
        # If a join order is provided, we can use all intermediates from the join order, which is way more efficient.
        #
        # We call the container candidate_joins, even though it also contains the base tables.

        if join_order is not None:
            candidate_joins = [node.tables() for node in join_order.iternodes()]
        else:
            candidate_joins = pb.util.powerset(query.tables())

        for join in candidate_joins:
            if not join:
                # skip the empty set
                continue

            estimated_cardinality = self.calculate_estimate(query, join)
            if math.isnan(estimated_cardinality):
                # Make sure to check for *NaN*, rather than just true-ish value. 0 is a valid cardinality estimate!
                warnings.warn(
                    f"Could not estimate cardinality for intermediate {join} in query {query}."
                )
                continue

            cardinalities.add_cardinality(join, estimated_cardinality)

        return cardinalities

    def describe(self) -> pb.util.jsondict:
        return {"name": "random_cardinalities"}


# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = pb.postgres.connect()
job_workload = pb.workloads.job()

# Now let's generate the optimization pipeline with our new cardinality estimator
pipeline = pb.MultiStageOptimizationPipeline(postgres_db)
pipeline.setup_plan_parameterization(
    JitteringCardinalityEstimator(postgres_db.optimizer())
)
pipeline.build()

# Run a couple of optimizations. Hopefully, they each contain a different query plan with different cardinality estimates
query = job_workload["1a"]
for i in range(3):
    print("Run", i + 1, "::")
    optimized_query = pipeline.optimize_query(query)
    query_plan = postgres_db.optimizer().query_plan(optimized_query)
    print(query_plan.inspect())
    print("--- --- ---")

# Depending on your Postgres installation, you may notice a caveat:
#
# For setups with pg_hint_plan, the estimates of the base tables do not change. This is because pg_hint_plan can only hint
# cardinalities for intermediate results/joins. Therefore, even though our estimation algorithm produced different estimates
# for the base tables, the hinting backend had to ignore these estimates in the final query.
# If you take a look at the hint block, you will notice that cardinality hints for base tables indeed do not exist. Disabling
# the warnings-filter in line 22 will also show warnings during the query generation.
#
# On the other hand, if you use pg_lab (https://github.com/rbergm/pg_lab), we can hint cardinalities for base tables just fine,
# so there are no issues.
#
# The Postgres interface tries to detect the available hinting backend automatically and uses the appropriate hinting features.
# However, this requires that the database system runs in the exact same namespace as PostBOUND (e.g. in the same Docker
# container). Otherwise, you might need to specify the hinting backend manually.
