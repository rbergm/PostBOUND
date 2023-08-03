# PostBOUND

This directory contains the Python implementation of our PostBOUND framework. The code itself is located in
the `postbound` directory. PostBOUND is an optimization framework for SQL queries that enables an easy implementation
of optimization strategies as well as a transparent evaluation of different optimization settings. It is implemented
as a Python tool that takes an input query, applies a user-configured optimization pipeline to that query and produces
an annotated output query that enforces the selected query plan during execution on a native database system.


## Getting started

All package requirements can be installed from the `requirements.txt` file. To use PostBOUND in different projects,
it can also be build and installed as a local package using pip. See the section below for details.

In addition to the Python packages, PostBOUND also needs a database connection in order to optimize and execute queries.
Currently, PostgreSQL and MySQL are supported, with the MySQL features being a bit more limited due to restrictions of
the system. The root directory of the PostBOUND repository contains setup utilities for some database systems, databases
and workloads.

The best way to familiarize yourself with PostBOUND is to study the examples and the documentation of the used classes
and functions. A high-level documentation is also being worked on, but still subject to change and not entirely
up-to-date. Therefore, the Python documentation of the source code is more extensive. Consult it for the specifics of _how_ to
use a specific feature and take a look at the examples to get an idea of _when_ to use it and which features are available. The
best starting point for the in-code documentation is the `__init__.py` file in the `postbound` source directory.

We also published a paper[^1] which explains the concepts that motivated the initial versions of PostBOUND.
Notice however, that at the time of its publication the framework had a much more limited scope and was heavily
expanded since then. More specifically, PostBOUND is no longer limited to upper bound-driven optimization strategies
and much more independent of specific database systems.


## Example

The following snippet gives a glimpse of the different parts of the framework and how they can interact. The specific
example implements the UES upper-bound optimization algorithm[^2] to obtain an optimized join order for the queries of
the Join Order Benchmark[^3] and applies them to a Postgres database instance.

```python
##
## Step 0: imports
##

# optimization modules
from postbound import postbound as pb
from postbound.optimizer import presets

# database modules
from postbound.db import postgres

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
optimization_pipeline = pb.TwoStageOptimizationPipeline(postgres_instance)
optimization_pipeline.load_settings(ues_settings)
optimization_pipeline.build()

##
## Step 3: Query optimization
##
input_query = job_workload["1a"]
optimized_query = optimization_pipeline.optimize_query(input_query)

##
## Step 4: Query execution
##
query_result = postgres_instance.execute_query(optimized_query)
print(query_result)
```

This examples can be *almost* executed as-is, there is only one setup step missing: in Step 1, PostBOUND is asked to connect to
a Postgres database, but no information is provided on how this connection should be obtained. By default, PostBOUND reads this
information from a hidden config file. In the case of Postgres, this file is called `.psycopg_connection` and it has to be
placed in the same directory from which the code is executed. The Postgres file has to contain a connection string that can
be used to establish a database connection, such as `dbname=<my db> user=<my user> host=localhost`. Consult the documentation
on the database systems for more info on what information is required and how it should be stored. As a final note, the
database setup utilities that are shipped with PostBOUND also contain scripts that automatically generate valid connect files
for the system. These files than only need to be moved to the correct location.


## Package structure

The `postbound` directory contains the actual source code of the framework. On a high-level, the PostBOUND framework is
structured as follows:

| Package       | Description |
|---------------|-------------|
| `optimizer`   | provides the different optimization strategies, interfaces and some pre-defined algorithms |
| `qal`         | provides the query abstraction used throughout PostBOUND, as well as logic to parse and transform query instances |
| `db`          | contains all parts of PostBOUND that concern database interaction. That includes retrieving data from different database systems, as well as generating optimized queries to execute on the database system |
| `experiments` | provides tools to conveniently load benchmarks and to measure their execution time for different optimization settings |
| `util`        | contains algorithms and types that do not belong to specific parts of PostBOUND and are more general in nature |
| `vis`         | contains utilities to visualize different concepts in query optimization (join orders, join graphs, query execution plans, ...) |

The actual optimization pipelines is defined in the `postbound` module at the package root. Depending on the specific
use-case, different pipelines are available.

In addition to the framework implementation, a couple of extra directories are provided. They contain the following:

| Directory  | Contents |
|------------|----------|
| `analysis` | contains Jupyter notebooks that contain tests for third-party libraries that are used by PostBOUND. |
| `docs`     | contains the high-level documentation as well as infrastructure to export the source code documentation |
| `examples` | contains general examples for typical usage scenarios. These should be run from the root directory, e.g. as `python3 examples/example-01-basic-workflow.py` |
| `tests`    | contains the unit tests and integration tests for the framework implementatino. These should also be run from the root directory, e.g. as `python3 -m unittest tests` |


## Installation as a Python module

PostBOUND provides (work in progress) support for packaging the entire source code into a `wheel` file that can be
installed using pip.

You can create the archive with the `python3 -m build` command, followed by an invocation of `pip install <wheel file>`
in the `dist` directory.


---

## Literature

[^1] Bergmann et al.: "PostBOUND: PostgreSQL with Upper Bound SPJ Query Optimization", BTW'2023 ([paper](https://dl.gi.de/handle/20.500.12116/40318))
[^2] Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'21 ([paper](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper01.pdf), [GitHub](https://github.com/axhertz/SimplicityDoneRight))
[^3] Leis et al.: "How Good are Query Optimizers, Really?", PVLDB'15 ([paper](https://dl.acm.org/doi/10.14778/2850583.2850594))
