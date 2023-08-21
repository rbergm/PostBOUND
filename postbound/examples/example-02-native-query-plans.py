#!/usr/bin/env python3

from postbound import postbound as pb
from postbound.db import db, postgres
from postbound.qal import qal, formatter
from postbound.optimizer import jointree, stages
from postbound.optimizer.strategies import native
from postbound.experiments import workloads

# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = postgres.connect()
job_workload = workloads.job()

# Since obtaining native execution plans is a pretty common use-case, there already is a pre-defined strategy to do this.
# Take a look at the native module for other strategies.
# If we were to use a different database in our NativeOptimizer, we would optimize queries using that database, but execute
# them on a different system
predef_pipeline = pb.IntegratedOptimizationPipeline(postgres_db)
predef_pipeline.optimization_algorithm = native.NativeOptimizer(postgres_db)


# Nevertheless, native optimization (or parts of it) can still be implemented using only a couple lines of code:
class OurNativeOptimizer(stages.CompleteOptimizationAlgorithm):
    def __init__(self, optimizer: db.OptimizerInterface) -> None:
        super().__init__()
        self.optimizer = optimizer

    def optimize_query(self, query: qal.SqlQuery) -> jointree.PhysicalQueryPlan:
        # Obtain the native query exection plan
        native_plan = self.optimizer.query_plan(query)

        # Generate the optimizer information for the plan.
        # Notice the distinction between an execution plan as modelled by the database interface, and the execution plan as
        # used by the optimization strategies
        execution_plan = jointree.PhysicalQueryPlan.load_from_query_plan(native_plan)
        return execution_plan

    def describe(self) -> dict:
        return {"name": "native_plans"}


custom_pipeline = pb.IntegratedOptimizationPipeline(postgres_db)
custom_pipeline.optimization_algorithm = OurNativeOptimizer(postgres_db.optimizer())


# We can use both pipelines exactly as exepected and they should also provide the same results (except if the native optimizer
# changes its mind between two optimizations of the same query)
query = job_workload["1a"]

print("Pre-defined strategy:")
predef_optimization = predef_pipeline.optimize_query(query)
print(formatter.format_quick(predef_optimization))
print("--- --- ---")

print("Custom strategy:")
custom_optimization = custom_pipeline.optimize_query(query)
print(formatter.format_quick(custom_optimization))
print("--- --- ---")
