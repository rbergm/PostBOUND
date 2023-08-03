# PostBOUND

PostBOUND is a framework for studying query optimization algorithms for (relational) database systems. It provides
tools to easily implement prototypes of new optimization algorithms and to compare them in a transparent and
reproducible way. This repository provides the actual Python implementation of the PostBOUND framework (located in the
`postbound` directory), along with a number of utilities that automate some of the tedious parts of automating the
evaluation of an optimization algorithm on a specific benchmark. Those utilities are focused on setting up different
popular database management systems and loading commonly used databases and benchmarks for them.

## Overview

The repository is structured as follows. Most likely you will be interested in the `postbound` directory. Almost all of the subdirectories contain further READMEs that explain their purpose and structure in more detail.

| Folder        | Description |
| ------------- | ----------- |
| `postbound`   | Contains the source code of the PostBOUND framework |
| `python-3.10` | Contains utilities to setup Python version 3.10, which is currently the earliest Python version supported by PostBOUND |
| `postgres`, `mysql` and `oracle` | Contain utilities to setup instances of the respective database systems and contain system-specific scripts to import popular benchmarks for them |
| `workloads`   | Contains the raw SQL queries for some popular benchmarks |
| `util`        | Provides different other utilities that are not directly concerned with specific database systems, but rather with common problems encoutered when benchmarking query optimizers |
