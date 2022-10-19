#!/usr/bin/env python3

# make sure to run this example script from within the examples/ directory in order to have all paths setup correctly

import sys
from datetime import datetime

sys.path.append("../")
from transform import db, mosp, ues  # noqa: E402
from postgres import hint  # noqa: E402


# ===
# Load the input query from the JOB workload (supply a different query label as CLI argument)
input_query = sys.argv[1] if len(sys.argv) > 1 else "1a"
with open(f"../../workloads/JOB-Queries/implicit/{input_query}.sql", "r") as query_file:
    raw_query = " ".join(query_file.readlines())


# ===
# MospQuery is the central query abstraction used in all PostBOUND components
query = mosp.MospQuery.parse(raw_query)
print("Input query:")
print(query)
print()


# ===
# UES query optimization generates an optimized join order for the query. The concrete optimization behaviour can be
# controlled via many arguments.
optimized_query = ues.optimize_query(query)
print("Optimized query:")
print(optimized_query)
print()


# ===
# HintedMospQuery is the interface to generate query hints for an optimized query. The specific choice of hints
# is left to user-supplied strategies. Hints may either influence the Postgres planner behaviour for the entire query,
# or for individual joins and base table scans.
operator_hints = hint.HintedMospQuery(optimized_query)

# select the actual hints
operator_hints.set_pg_param(enable_nestloop="off")  # emulate UES setup: no NLJ
operator_hints.set_pg_param(join_collapse_limit=1)  # emulate UES setup: enforce correct join ordering

hinted_query = operator_hints.generate_query()  # use query_hints.generate_sqlcomment() to get just the hints
print("Operator hints:")
print(operator_hints.generate_sqlcomment())
print()


# ===
# Execute the query
database = db.DBSchema.get_instance()
start_time = datetime.now()
result = database.execute_query(hinted_query, cache_enabled=False)
end_time = datetime.now()
execution_time = end_time - start_time
print("Query result:", result, "|| Execution time:", execution_time)
