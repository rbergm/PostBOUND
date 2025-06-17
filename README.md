# PostBOUND

![GitHub License](https://img.shields.io/github/license/rbergm/PostBOUND)
![Static Badge](https://img.shields.io/badge/version-0.15.4-blue)

<p align="center">
    <img src="docs/figures/postbound-logo.svg" style="width: 150px; margin: 15px;">
</p>

PostBOUND is a Python framework for studying query optimization in database systems.
At a high level, PostBOUND has the following goals and features:

- 🏃‍♀️ **Rapid prototyping:** PostBOUND allows researchers to implement specific phases of the optimization process, such
  as the cardinality estimator or the join order optimization. Researchers can focus precisely on the parts that they are
  studying and use _reasonable_ defaults for the rest. See [🧑‍🏫 Example](#-example) for how this looks in practice.
- 🧰 **No boilerplate:** to ease the implementation of the optimizers, PostBOUND provides a large toolbox of
  support functionality, such as query parsing, join graph construction, relational algebra or database statistics.
- 📊 **Transparent benchmarks:** once a new optimizer prototype is completed, the benchmarking tools allow to compare this
  prototype against other optimization strategies in a transparent and reproducible way. As a core design principle,
  PostBOUND executes queries on an actual database system such as PostgreSQL, rather than research systems or artificial
  "lab" comparisons. See [💡 Essentials](#-essentials) for more information on this.
- 🔋 **Batteries included:** in addition to the Python package, PostBOUND provides a lot of utilities to setup databases
  and load commonly used benchmarks (e.g., JOB, Stats and Stack).

**| [💻 Installation](https://postbound.readthedocs.io/en/latest/setup.html) | [📖 Documentation](https://postbound.readthedocs.io/en/latest/) | [🧑‍🏫 Examples](https://github.com/rbergm/PostBOUND/tree/main/examples) |**


## ⚡️ Quick Start

The fastest way to get an installation of PostBOUND up and running is to use the provided Dockerfile.
You can build your Docker image with the following command:

```sh
docker build -t postbound \
    --build-arg TIMEZONE=$(cat /etc/timezone) \
    --build-arg USE_PGLAB=true \
    --build-arg OPTIMIZE_PG_CONFIG=true \
    --build-arg SETUP_STATS=true \
    --build-arg SETUP_JOB=false \
    --build-arg SETUP_STACK=false \
    .
```

This will create a Docker image with a local Postgres instance (using [pg_lab](https://github.com/rbergm/pg_lab)) and
setup the Stats, JOB and Stack benchmarks.
Adjust these arguments to your needs.
See [Essentials](#-essentials) for why the Postgres instance is necessary.
All supported build arguments are listed under [Docker options](#-docker-options).

Once the image is built, create a container like so:

```sh
docker run -dt \
    --shm-size 4G \
    --name postbound \
    --volume $PWD/postbound-docker:/postbound/public \
    --publish 5432:5432 \
    postbound
```

Adjust the amount of shared memory depending on your machine.

> [!TIP]
> Shared memory is used by Postgres for its internal caching and therefore paramount for good server performance.
> The general recommendation is to set it to at least 1/4 of the available RAM.

The Postgres server will be available at port 5432 from the host machine (using the user _postbound_ with the same
password).
The volume mountpoint can be used to easily copy experiment scripts into the container and to export results back out
again.

You can connect to the PostBOUND container using the usual

```sh
docker exec -it postbound /bin/bash
```

The shell enviroment is setup to have PostBOUND available in a fresh Python virtual environment (which is activated by
default).
Furthermore, all Postgres utilities are available in the _PATH_.

> [!TIP]
> If you want to install PostBOUND directly on your machine, the
> [documentation](https://postbound.readthedocs.io/en/latest/setup.html) provides a detailed setup guide.


## 💡 Essentials

As a central design decision, PostBOUND is not integrated into a specific database system.
Instead, it is implemented as a Python framework operating on top of a running database instance.
This decision was made to ensure that the optimization strategies are actually useful in practice - PostBOUND executes the
optimized queries on a real databases and we treat the execution time as the ultimate measure of optimization quality.

However, this decision means that we need a way to ensure that the optimization decisions made within the framework are
actually used when executing the query in the context of the target database.
This is achieved by using query hints which typically encode the optimization decisions in comment blocks within the query.

In the case of Postgres, this interaction roughly looks like this:

<p align="center">
  <img src="docs/figures/postbound-pg-interaction.svg" style="width: 600px; margin: 15px;">
</p>

Depending on the actual database system, these hints might differ in syntax as well as semantics.
Generally speaking, PostBOUND figures out which hints to use on its own, without user intervention.
Sadly, PostgreSQL does not support query hints out of the box.
Therefore, PostBOUND relies on either [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan) or
[pg_lab](https://github.com/rbergm/pg_lab) to provide the necessary hinting functionality.

> [!NOTE]
> PostBOUND's database interaction is designed to be independent of a specific system (such as PostgreSQL, Oracle, ...).
> However, the current implementation is most complete for PostgreSQL with limited support for MySQL.
> This is due to practical reasons, mostly our own time budget and the popularity of PostgreSQL in the optimizer research
> community.


## 🧑‍🏫 Example

The typical end-to-end workflow using PostBOUND looks like this:

1. **Implement your new optimization strategy**. To do so, you need to figure out which parts of the optimization process you
   want to customize and what the most appropriate optimization pipeline is. In a nutshell, the optimization pipeline is
   a mental model of how the optimizer works. Commonly used pipelines are the textbook-style pipeline (i.e. using plan
   enumerator, cost model and cardinality estimator), or the multi-stage pipeline which first computes a join order and
   afterwards selects the best physical operators. The pipeline determines which interfaces can be implemented.
2. Select your **target database system** and **benchmark** that should be used for the evaluation.
3. Optionally, select different optimization strategies that you want to compare against.
4. Use the **benchmarking tools** to execute the workload against the target database system.

For example, a random join order optimizer could be implemented like this:

```python
import random

import postbound as pb

# Step 1: define our optimization strategy.
# We use the pre-defined join graph to keep track of free tables.
class RandomJoinOrderOptimizer(pb.JoinOrderOptimization):
    def optimize_join_order(self, query: pb.SqlQuery) -> pb.LogicalJoinTree:
        join_tree = pb.LogicalJoinTree()
        join_graph = pb.opt.joingraph.JoinGraph(query)

        while join_graph.contains_free_tables():
            candidate_tables = [
                path.target_table for path in join_graph.available_join_paths()
            ]
            next_table = random.choice(candidate_tables)

            join_tree = join_tree.join_with(next_table)
            join_graph.mark_joined(next_table)

        return join_tree

    def describe(self) -> pb.util.jsondict:
        return {"name": "random-join-order"}

# Step 2: connect to the target database, load the workload and
# setup the optimization pipeline.
pg_imdb = pb.postgres.connect(config_file=".psycopg_connection_job")
job = pb.workloads.job()

optimization_pipeline = (
    pb.MultiStageOptimizationPipeline(pg_imdb)
    .setup_join_order_optimization(RandomJoinOrderOptimizer())
    .build()
)

# (Step 3): we just compare against the native Postgres optimizer,
# which does not require any additional setup.

# Step 4: execute the workload.
# We use the QueryPreparationService to prewarm the database buffer run all queries
# as EXPLAIN ANALYZE.
query_prep = pb.experiments.QueryPreparationService(
    prewarm=True, analyze=True, preparatory_statements=["SET geqo TO off;"]
)
native_results = pb.execute_workload(
    job, pg_imdb, query_preparation=query_prep, workload_repetitions=3
)
optimized_results = pb.optimize_and_execute_workload(
    job, optimization_pipeline, query_preparation=query_prep, workload_repetitions=3
)

pb.experiments.prepare_export(native_results).to_csv("job-results-native.csv")
pb.experiments.prepare_export(optimized_results).to_csv("job-results-optimized.csv")
```


## 🫶 Reference

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{bergmann2025elephant,
  author       = {Rico Bergmann and
                  Claudio Hartmann and
                  Dirk Habich and
                  Wolfgang Lehner},
  title        = {An Elephant Under the Microscope: Analyzing the Interaction of Optimizer
                  Components in PostgreSQL},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {3},
  number       = {1},
  pages        = {9:1--9:28},
  year         = {2025},
  url          = {https://doi.org/10.1145/3709659},
  doi          = {10.1145/3709659},
  timestamp    = {Tue, 01 Apr 2025 19:03:19 +0200},
  biburl       = {https://dblp.org/rec/journals/pacmmod/BergmannHHL25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

---


## 📖 Documentation

A detailed documentation of PostBOUND is available [here](https://postbound.readthedocs.io/en/latest/).


## 🐳 Docker options

| Argument | Allowed values | Description | Default |
|----------|----------------|-------------|---------|
| `TIMEZONE` | Any valid timezone identifier | Timezone of the Docker container (and hence the Postgres server). It is probably best to just use the value of `cat /etc/timezone` | `UTC` |
| `USERNAME` | Any valid UNIX username. | The username within the Docker container. This will also be the Postgres user and password. | `postbound` |
| `SETUP_IMDB` | `true` or `false` | Whether an [IMDB](https://doi.org/10.14778/2850583.2850594) instance should be created as part of the Postgres setup. PostBOUND can connect to the database using the `.psycopg_connection_job` config file. | `false` |
| `SETUP_STATS` | `true` or `false` | Whether a [Stats](https://doi.org/10.14778/3503585.3503586) instance should be created as part of the Postgres setup. PostBOUND can connect to the database using the `.psycopg_connection_stats` config file. | `false` |
| `SETUP_STACK` | `true` or `false`| Whether a [Stack](https://doi.org/10.1145/3448016.3452838) instance should be created as part of the Postgres setup. PostBOUND can connect to the database using the `.psycopg_connection_stack` config file. | `false` |
| `OPTIMIZE_PG_CONFIG` |  `true` or `false` | Whether the Postgres configuration parameters should be automatically set based on your hardware platform. Rules are based on [PGTune](https://pgtune.leopard.in.ua/) by [le0pard](https://github.com/le0pard). | `false` |
| `PG_DISK_TYPE` | `SSD` or `HDD` | In case the Postgres server is automatically configured (see `OPTIMIZE_PG_CONFIG`) this indicates the kind of storage for the actual database. In turn, this influences the relative cost of sequential access and index-based access for the query optimizer. | `SSD` |
| `PG_VER` | 16, 17, ... | The Postgres version to use. Notice that pg_lab supports fewer versions. This value is passed to the `postgres-setup.sh` script of the Postgres tooling (either under `db-support` or from pg_lab), which provides the most up to date list of supported versions. | 17 |
| `USE_PGLAB` | `true` or `false` | Whether to initialize a [pg_lab](https://github.com/rbergm/pg_lab) server instead of a normal Postgres server. pg_lab provides advanced hinting capabilities and offers additional extension points for the query optimizer. | `false` |

The PostBOUND source code is located at `/postbound`. If pg_lab is being used, the corresponding files are located at `/pg_lab`.
The container automatically exposes the Postgres port 5432 and provides a volume mountpoint at `/postbound/public`. This
mountpoint can be used to easily get experiment scripts into the container and to export results back out again.

> [!TIP]
> pg_lab provides advanced hinting support (e.g. for materialization or cardinality hints for base tables) and offers
> additional extension points for the query optimizer (e.g. hooks for the different cost functions).
> If pg_lab is not used, the Postgres server will setup pg_hint_plan instead.

---

## 📑 Repo Structure

The repository is structured as follows.
The `postbound` directory contains the actual source code, all other folders are concerned with "supporting" aspects
(which are nevertheless important..).
Almost all of the subdirectories contain further READMEs that explain their purpose and structure in more detail.

| Folder        | Description |
| ------------- | ----------- |
| `postbound`   | Contains the source code of the PostBOUND framework |
| `docs`        | contains the high-level documentation as well as infrastructure to export the source code documentation |
| `examples`    | contains general examples for typical usage scenarios. These should be run from the root directory, e.g. as `python3 -m examples.example-01-basic-workflow` |
| `tests`       | contains the unit tests and integration tests for the framework implementatino. These should also be run from the root directory, e.g. as `python3 -m unittest tests` |
| `db-support`  | Contains utilities to setup instances of the respective database systems and contain system-specific scripts to import popular benchmarks for them |
| `workloads`   | Contains the raw SQL queries of some popular benchmarks |
| `tools`       | Provides different other utilities that are not directly concerned with specific database systems, but rather with common problems encoutered when benchmarking query optimizers |


## 📑 Framework Structure

The `postbound` directory contains the actual source code of the framework.
On a high-level, the PostBOUND framework is structured as follows:

![Overview of the major PostBOUND packages](docs/figures/postbound-package-overview.png)

| Package       | Description |
|---------------|-------------|
| `optimizer`   | provides the different optimization strategies, interfaces and some pre-defined algorithms |
| `qal`         | provides the query abstraction used throughout PostBOUND, as well as logic to parse and transform query instances |
| `db`          | contains all parts of PostBOUND that concern database interaction. That includes retrieving data from different database systems, as well as generating optimized queries to execute on the database system |
| `experiments` | provides tools to conveniently load benchmarks and to measure their execution time for different optimization settings |
| `util`        | contains algorithms and types that do not belong to specific parts of PostBOUND and are more general in nature |
| `vis`         | contains utilities to visualize different concepts in query optimization (join orders, join graphs, query execution plans, ...) |

The actual optimization pipelines are defined in the `postbound` module at the package root.
Depending on the specific use-case, different pipelines are available.

> [!TIP]
> The [Documentation](https://postbound.readthedocs.io/en/latest/) provides a more detailed overview of the different packages and their purpose.
