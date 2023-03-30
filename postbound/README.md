# PostBOUND

This directory contains the Python implementation of our PostBOUND framework. The code itself is located in
the `postbound` directory. PostBOUND is an optimization framework for SQL queries that enables an easy implementation
of optimization strategies as well as a transparent evaluation of different optimization settings. It is implemented
as a Python tool that takes an input query, applies a user-defined optimization pipeline to that query and produces an
annotated output query that enforces the selected query plan during execution of the query on a native database system.

## Getting started

All package requirements can be installed from the `requirements.txt` file. In addition to the Python packages,
PostBOUND also needs a database connection in order to optimize and execute queries. How this connection is obtained,
depends on the specific implementation of the database system (PostBOUND only talks to an abstract interface that is
implemented by different systems). The root directory of the PostBOUND repository contains setup utilities for some
database systems, databases and workloads.

The best way to familiarize yourself with PostBOUND is to study the examples and the documentation of the used classes
and functions. At some point in the future we might also create a high-level documentation, but for now we only provide
the in-code documentation. Therefore, this documentation is oftentimes pretty extensive. Take a look at the
documentation of the `postbound` package to get started.

You can also take a look at our paper[^0] which explains the concepts that motivated the early versions of PostBOUND.
However, since then we worked hard on generalizing and expanding the functionality and features of the framework.

```python
##
## Step 0: imports
##

# optimization modules
from postbound import postbound as pb
from postbound.optimizer import presets

# database modules
from postbound.db import postgres
from postbound.db.systems import systems

# workload modules
from postbound.experiments import workloads

##
## Step 1: System setup
##
postgres_instance = postgres.connect()
presets.apply_standard_system_options()
job_workload = workloads.job()
ues_settings = presets.fetch("ues")

##
## Step 2: Optimization pipeline setup
##
optimization_pipeline = pb.OptimizationPipeline(systems.Postgres(postgres_instance))
optimization_pipeline.load_settings(ues_settings)
optimization_pipeline.build()

##
## Step 3: Query optimization
##
optimized_query = optimization_pipeline.optimize_query(job_workload["1a"])

##
## Step 4: Query execution
##
query_result = postgres_instance.execute_query(optimized_query)
print(query_result)
```

## Package structure

On a high-level, the PostBOUND framework is structured as follows:

| Package     | Description                                                                                                                                                                                                 |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `optimizer` | provides the different optimization strategies, interfaces and some pre-defined algorithms                                                                                                                  |
| `qal`       | provides the query abstraction used throughout PostBOUND, as well as logic to parse and transform query instances                                                                                           |
| `db`        | contains all parts of PostBOUND that concern database interaction. That includes retrieving data from different database systems, as well as generating optimized queries to execute on the database system |
| `util`      | contains algorithms and types that do not belong to specific parts of PostBOUND and are more general in nature                                                                                              |                                                                                                                                                                                                             |

The actual optimization pipeline is defined in the `postbound` module at the package root.

---

## Literature

[^0] Bergmann et al.: "PostBOUND: PostgreSQL with Upper Bound SPJ Query Optimization", BTW'2023 ([paper](https://dl.gi.de/handle/20.500.12116/40318))
