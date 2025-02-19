#!/usr/bin/env python3
#
# This script shows how to implement a custom join ordering algorithm by implementing a simple randomized "optimizer"
# The algorithm only produces linear join paths for queries without cross products.
#
# Requirements: a running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities in the root postgres directory.
#

import random
import warnings
from typing import Optional

import postbound as pb
from postbound.optimizer import joingraph, validation

warnings.simplefilter("ignore")


class RandomJoinOrderOptimizer(pb.JoinOrderOptimization):
    # The entire join ordering algorithm is implemented in this class. It satisfies the interface of the corresponding
    # optimization stage.

    def __init__(self, target_db: Optional[pb.Database] = None) -> None:
        super().__init__()

        # Strictly speaking, we do not need the target database here, since we are only concerned with linear join orders.
        # However, we might also develop a more adaptive algorithm that checks which hints are supported by the target
        # database. If it also supports bushy join orders, we might generate such join orders instead.
        # Therefore, we store the target database here and demonstrate how we can easily access it using the DatabasePool
        self.target_db = target_db if target_db is not None else pb.db.current_database()

    def optimize_join_order(self, query: pb.SqlQuery) -> pb.opt.LogicalJoinTree:
        # This is the most important method that handles the actual join order optimization

        # In our optimizer we must maintain two data structures:
        # The join graph stores which tables have already been joined and which joins are available next
        # The join tree stores the join order that we have constructed so far
        join_graph = joingraph.JoinGraph(query)
        join_tree = pb.opt.LogicalJoinTree.empty()

        # Our algorithm simply joins one table after another, until all tables have inserted into the join order
        while join_graph.contains_free_tables():

            # Figure out which joins are available right now. In an actual optimization scenario, these would be candidate
            # joins and the algorithm would need to figure out which join is the best one.
            candidate_joins = join_graph.available_join_paths()
            candidate_tables = [path.target_table for path in candidate_joins]

            # For our algorithm, we simply chosse the next table at random
            next_table = random.choice(candidate_tables)

            # Update our data to store the optimization progress.
            # Notice that join trees are immutable, hence a new instance is produced. On the other hand, join graphs can be
            # updated directly
            join_tree = join_tree.join_with(next_table)
            join_graph.mark_joined(next_table)

        return join_tree

    def describe(self) -> dict:
        return {"name": "random_linear"}

    def pre_check(self) -> validation.OptimizationPreCheck:
        # Our optimizer only works for queries without cross products (there are no join paths between the different
        # subqueries). Therefore, we must require that it is not executed on "bad" queries via a pre-check.
        query_check = validation.CrossProductPreCheck()

        # Furthermore, we must ensure that the target database supports enforcing linear join orders. This is exactly, what
        # the SupportedHintCheck does.
        db_check = validation.SupportedHintCheck(pb.opt.HintType.LinearJoinOrder)

        # No, just combine the individual checks into a single one check
        return validation.CompoundCheck([query_check, db_check])


# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = pb.postgres.connect()
job_workload = pb.workloads.job()

# Now let's generate the optimization pipeline with our new optimizer
pipeline = pb.TwoStageOptimizationPipeline(postgres_db)
pipeline.setup_join_order_optimization(RandomJoinOrderOptimizer(postgres_db))
pipeline.build()

# Run a couple of optimizations. Hopefully, they each contain a different join order
query = job_workload["1a"]
for i in range(3):
    print("Run", i+1, "::")
    optimized_query = pipeline.optimize_query(query)
    query_plan = postgres_db.optimizer().query_plan(optimized_query)
    print(query_plan.inspect(skip_intermediates=True))
    print("--- --- ---")
