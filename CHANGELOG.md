# Changelog

Version numbers are composed of three components, i.e. _major_._minor_._patch_
As a rough guideline, patch releases are just for fixing bugs or adding minor details (e.g. a new default parameter to some
function), minor releases change slightly larger parts of the framework or add significant new functionality (e.g. a new
optimization pipeline or support for an SQL feature). Major releases fundamentally shift how the framework is used and indicate
stability. Since we are not ready for the 1.0 release yet, this does not matter right now.

The [history](HISTORY.md) contains the changelogs of older PostBOUND releases.

---

# Version 0.20.2

This is an incremental updating that polishes some rough edges and fixes a couple bugs.

## 📰 Updates
- Improved the control flow of the Postgres server setup during Docker initialization.
- Miscellaneous improvements to the type hints across the entire codebase. We will continue this work in the upcoming releases.

## 🏥 Fixes
- Fixed the Postgres timeout executor always creating two new connections to the database server instead of just one for the
  actual query execution.
- Fixed a corner case in the Postgres timeout handling when the timeout worker could not establish a connection to the
  database.
- Fixed closing a Postgres connection leaving the timeout watcher connection open.
- Fixed discrepancy between JSON export format of `ColumnReference` and the expected format during loading
- Fixed SQL parser sometimes still representing ANY/ALL predicates as function expressions instead of the new quantifier
  expressions.

---


# Version 0.20.1

This is just a small bug fix/patch release with some minor additions.

## 🐣 New features
- `as_predicate()` now supports string representations of the operators for better usability. E.g., you can now do
  `as_predicate(my_col, "=", 5)` instead of `as_predicate(my_col, LogicalOperator.Equal, 5)`. The old way still works is not
  intended to be removed.
- Added a `standard_logger()` utility for consistent logging output across all modules

## 💀 Breaking changes
- Technically the change to the `tables()` method of `DatabaseSchema` is breaking, but the method was already documented to
  return all user-defined tables.

## 📰 Updates
- `DatabaseSchema.tables()` now ignores system tables by default

## 🏥 Fixes
- Fixed internal storage layout of `SqlQuery`
- Fixed type error in `OrderBy.create_for()`
- Fixed `PhysicalOperatorAssignment` method chaining not working correctly

## ⚠️ Deprecations
- The _ues_, _tonic_, _presets_, and _experiments_ module are now deprecated and will be moved to the separate optimizer
  repository for version 0.21.0.

## 🪲 Known bugs
- The automatic optimization of the Postgres server configuration as part of the Docker installation does not work
  on MacOS. Currently, this should be considered as wontfix.

---


# Version 0.20.0

This release conducts some major changes of the framework structure, making the different parts more accessible and more
visible. Sadly, this comes with a few breaking changes. In particular, we perform the following high-level updates:

1. Almost all of the framework is now accessible with just one module indirection, i.e. using `pb.module.function` instead
   of the old `pb.module1.module2.module3.function`.
2. This flat structure allows for consistent lazy-loading of all modules to improve overall responsiveness
3. We transition to [quacklab-python](https://github.com/rbergm/quacklab-python) as the new backend for hinting-aware
   DuckDB. This change was required due to upstream updates of the DuckDB packaging that broke our previous implementation.
   We use this opportunity to also rename the dependency from _duckdb_ to _quacklab_. This has the nice property of
   erroring if our backend cannot find a DuckDB installation with hinting support.

At the same time, we use this release to prepare for a (perhaps final) major restructuring that we plan for v0.21.0:
moving existing optimizers from the core framework into a separate optimizer repository. This repository will be used as
a collection of interesting ideas in query optimization and will help to keep the core framework lean (especially by
preventing an explosion of dependencies). This will affect the UES and TONIC optimizers as well as the query generators.

## 🐣 New features
- Consistently use lazy-loading for all modules.
- The `postgres.connect()` method now accepts many additional file formats (TOML, JSON, YAML) for the configuration files.
- Overriding the  `describe()` method is no longer required for all optimization stages. If not implemented, a minimal
  default description is provided.
- Added `use()` methods to all optimization pipelines that did not have them yet.
- Added a new `scale_cardinality()` to query plans to scale estimated and/or actual cardinalities by a given factor.
- Extended SQL support: we can now parse and represent simple CASE expressions
  (_CASE R.a WHEN 1 THEN 'one' WHEN 2 THEN 'two'_), _IS DISTINCT FROM_ predicates and _ANY_/_ALL_ predicates.
- Calling `execute_workload()` with progressive output now automatically exports the experiment configuration to the output
  directory for better reproducibility. This is our first step towards an lab notebook-like experiment management.
- `PhysicalOperatorAssignment` and `PlanParameterization` now support method chaining for better usability.

## 💀 Breaking changes
- Transition to [quacklab-python](https://github.com/rbergm/quacklab-python) as the new backend for hinting-aware DuckDB
- All hinting-related data structures (e.g., `PhysicalOperatorAssignment`) are now available as top-level types.
- The query abstraction layer now exclusively focuses on query representation:
  - The query parser is now its own module (`postbound.parser`) instead of part of the qal. This solves all dependency
    issues between the query abstraction and the database schema in a very nice way. `parse_query()` is still available as
    a top-level function.
  - The query transformation tools are now its own module `postbound.transform` instead of part of the qal.
  - The relation algebra representation is now its own module `postbound.relalg` instead of part of the qal.
- All hinting-related data structures (e.g., `PhysicalOperatorAssignment`) are now available as top-level types.
- The optimizer module is now always called _opt_, i.e. the following does no longer work:
  `from postbound import optimizer` while calling `pb.opt` directly remains unchanged.
- Removed the `DBCatalog` type. This was only used to circumvent cyclic import issues with the `DatabaseSchema` between
  qal (parser) and the database module. Since the parser is now its own module, this is no longer necessary.
- The Postgres and DuckDB backends are now available as top-level modules.
- The visualization module is now available as a top-level module.

## 📰 Updates
- `Workload` is now a top-level type (in addition to being available in the `postbound.workloads` module).
- `Workload.read()` now accepts many additional options tailored for loading large workloads
- Added a *str* and *repr* method to `QueryPreparation` for better debugging support.
- Virtual tables are no longer included in database prewarming for Postgres.
- Improved methods to manually create _FROM_ and _ORDER BY_ clauses.

## 🏥 Fixes
- Rewrote the entire query execution with timeouts for Postgres. This should (hopefully) fix any remaining corner cases
  that previoulsy resulted in deadlocks.
- Fixed EXPLAIN ANALYZE plans that were extracted from Postgres and converted to a query plan not containing the correct
  actual cardinalities when parallel workers were used.
- Fixed handling of infinite date/timestamp values in the Postgres backend
- The _postgres-setup.sh_ script now actually uses PG 12.4 if that version was requested.

## ⚠️ Deprecations
- The _ues_, _tonic_, _presets_, and _experiments_ module are now deprecated and will be moved to the separate optimizer
  repository for version 0.21.0.

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
