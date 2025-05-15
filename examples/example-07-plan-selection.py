#!/usr/bin/env python3
#
# This script demonstrates how a BAO-style optimizer can be implemented using PostBOUND. For a given query, a number of
# candidate plans are generated and the best one is selected. To keep things simple, we do not use a learned model to select
# the optimal plan, but simply select the plan with the lowest estimated cost.
#
# Requirements: a running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities from db-support/postgres.
#

from typing import Optional

import postbound as pb


class PlanSelection(pb.CompleteOptimizationAlgorithm):
    # The entire optimization algorithm is implemented in this class. It satisfies the interface of the corresponding
    # optimization stage.

    def __init__(self, target_db: Optional[pb.postgres.PostgresInterface]) -> None:
        super().__init__()
        self._target_db = target_db or pb.db.DatabasePool.get_instance().current_database()

        # We need the default server configuration to make sure that our connection is in a clean state whenever we obtain a
        # new query plan. See the comment in _reset_db_state for more details.
        self._default_config = self._target_db.current_configuration(runtime_changeable_only=True)

        # These are all hints that we are going to use to generate different query plans. There are more hints available, but
        # we don't use them for simplicity (and because we would suffer from combinatorial explosion).
        self._hints: list[pb.opt.PhysicalOperator] = [pb.opt.ScanOperator.SequentialScan, pb.opt.ScanOperator.IndexScan,
                                                      pb.opt.JoinOperator.NestedLoopJoin, pb.opt.JoinOperator.HashJoin]

    def optimize_query(self, query: pb.SqlQuery) -> pb.QueryPlan:
        # This is the only real method that we need to implement. It should generate the entire query plan for the input query.

        best_plan: pb.QueryPlan = None

        # To implement our optimizer, we need to think about two questions::
        # 1. how do we generate different candidate plans, and
        # 2. how do we select the best one among the candidates
        #
        # For 1., we use a mechanism similar to BAO: we use hints to restrict the optimizer's search space, thereby obtaining
        # different plan. Our algorithm works as follows: from the set of all hints, we generate all possible true subsets.
        # Each subset indicates the set of operators that should be disabled.
        #
        # For 2., we simply re-use the cost model of the database system and select the plan with the lowest estimated cost.

        for disabled_operators in pb.util.collections.powerset(self._hints):
            if not disabled_operators or len(disabled_operators) == len(self._hints):
                continue

            # We need to be a bit careful with state management when using global settings. See the comment in _reset_db_state
            # for more details.
            self._reset_db_state()

            # We store the operators to disable in the assignment
            assignment = pb.PhysicalOperatorAssignment()
            for operator in disabled_operators:
                assignment.set_operator_enabled_globally(operator, False)

            # Once we obtained the operator configuration, we generate the corresponding query plan.
            # This works using the normal hinting mechanism of PostBOUND, which essentially captures the elements of the
            # assignment and integrates them into the query.
            hinted_query = self._target_db.hinting().generate_hints(query, physical_operators=assignment)
            query_plan = self._target_db.optimizer().query_plan(hinted_query)

            if best_plan is None or query_plan.estimated_cost < best_plan.estimated_cost:
                best_plan = query_plan

        if best_plan is None:
            raise ValueError(f"No valid plan found for query {query}")
        return best_plan

    def describe(self) -> pb.util.jsondict:
        return {
            "type": "plan_selection",
            "hints": self._hints,
        }

    def _reset_db_state(self) -> None:
        # Global settings are acutally global, i.e. they might also affect other queries running in the same session.
        # Therefore, we need to be a bit careful when using these and make sure to reset them after we are done.
        # Otherwise, the configuration from the first query plan is going to influence the construction of the second one, etc.
        self._target_db.apply_configuration(self._default_config)


# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = pb.postgres.connect()
job_workload = pb.workloads.job()

# Configure the optimization pipeline with our plan selection algorithm
pipeline = (pb.IntegratedOptimizationPipeline(postgres_db)
            .setup_optimization_algorithm(PlanSelection(postgres_db))
            .build())

# Run a couple of optimizations.
for label, query in job_workload.first(5).items():
    print(label, "::")
    optimized_plan = pipeline.query_execution_plan(query)
    print(optimized_plan.inspect(), end="\n\n")
