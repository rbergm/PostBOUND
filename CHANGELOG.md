# Changelog

Version numbers are composed of three components, i.e. _major_._minor_._patch_
As a rough guideline, patch releases are just for fixing bugs or adding minor details (e.g. a new default parameter to some
function), minor releases change slightly larger parts of the framework or add significant new functionality (e.g. a new
optimization pipeline or support for an SQL feature). Major releases fundamentally shift how the framework is used and indicate
stability. Since we are not ready for the 1.0 release yet, this does not matter right now.

The [history](HISTORY.md) contains the changelogs of older PostBOUND releases.

---

# Version 0.21.0

## 🐣 New features
- Database statistics now also provide histograms.
- The `OptimizerInterface` (e.g. `pg_instance.optimizer()`) now provides a method to parse an existing system-specific
  query plan to the generalized `QueryPlan`. This can be used as follows:

  ```python
  explain_query = pb.transform.as_explain(query)
  raw_plan = database.execute_query(explain_query)
  plan = database.optimizer().parse_plan(raw_plan)
  ```
- `execute_workload()` now supports many new output formats for writing the progressive output.
  Currently supported are: CSV, Parquet, JSON, HDF

## 📰 Updates
- `DatabaseStatistics.most_common_values()` now returns an actual `MostCommonValues` object instead of a list of tuples.
  The `MostCommonValues` can be used as a drop-in replacement for the old tuple-based API. In addition, it provides more
  high-level methods for working with the most common values.
- Enabled the MySQL and DuckDB backends to fall back to emulated statistics if the database does not provide them.
- The `QueryPreparation` API now provides the `projection` and `output` parameters to modify the *SELECT* clause and the type
  of results to gather for all queries in a more flexible and intuitive way (how did *explain=True*  and *analyze=True*
  interact?).
  The old API using *analyze=True*, etc. is now deprecated in favor of these new parameters.
- The `to_json()` and `to_json_dump()` utilities now support dataclasses out-of-the-box.

## 🏥 Fixes
- Fixed output format of the benchmarking log if additional entries are appended to an existing log. Essentially, we fix such
  entries being escaped twice.

## 💀 Breaking changes
- _None_

## ⚠️ Deprecations
- The old `QueryPrepration` API using *analyze=True*, etc. is now deprecated in favor of the more flexible *projection* and
  *output*  parameters. However, we currently have no plans to remove the old API.

## 🪲 Known bugs
- The automatic optimization of the Postgres server configuration as part of the Docker installation does not work
  on MacOS. Currently, this should be considered as wontfix.

---


# 🛣 Roadmap

Currently, we plan to implement the following features in the future (in no particular order):

- Providing a Substrait export for query plans
- Better benchmarking setup, mostly focused on comparing one or multiple optimization pipelines and creating better experiment
  logs and the ability to cancel/resume long-running benchmarks
- Adding popular optimization algorithms to the collection of pre-defined optimizers
