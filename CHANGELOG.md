# Changelog

Version numbers are composed of three components, i.e. _major_._minor_._patch_
As a rough guideline, patch releases are just for fixing bugs or adding minor details
(e.g. a new default parameter to some function), minor releases change slightly
larger parts of the framework or add significant new functionality (e.g. a new
optimization pipeline or support for an SQL feature). Major releases fundamentally
shift how the framework is used and indicate stability. Since we are not ready
for the 1.0 release yet, this does not matter right now.

The [history](HISTORY.md) contains the changelogs of older PostBOUND releases.

---

## Version 0.23.0

This release focuses on improvements of the database interface, especially the schema.
In addition, minor quality-of-life improvements throughout the framework should
make it more convenient overall.

## 🐣 New features

- The `DatabaseSchema` now supports composite keys and indexes. Sadly, this change
  introduces some breaking changes (see below).
- Introduced proper `PrimaryKeyRef` and `ForeignKeyRef` classes to easily work
  with primary keys/foreign keys, including those that consist of multiple columns.
- `ColumnInfo` now has a new `unique` field
- `Histogram` now provides explicit `bound-upper` and `bound-lower` interpolation
  modes.
- `SimpleJoin` now has a `key_of()` function to easily retrieve the join key of
  a given table.

## 💀 Breaking changes

- Removed the legacy MySQL backend.
- `DatabaseSchema.primary_key_column()` has been removed in favor of `primary_key_of()`.
  This also affects `TableInfo`
- `DatabaseSchema.foreign_keys_on()` now provides explicit `ForeignKeyRef` objects.
  This also affects the schema graph.
- Removed the default values for the `DatabaseSchema` constructor. We should be
  explicit about this stuff to prevent subtle bugs.
- Removed the `bound` interpolation mode of `Histogram`. Use the more explicit
  `bound-upper` or `bound-lower` instead.

## 📰 Updates

- NULLs are now treated consistenly across all statistics backends. Most importantly,
  all methods of the `StatisticsCatalog` (except for `null_frac`) exclude NULLs
  their calculations. This is in line with the behavior of PostgreSQL and the strategy
  that makes most sense with respect to the NULL semantics in SQL.
- `JoinGraph` is now an explicit type alias for [`nx.Graph`](https://networkx.org/documentation/stable/reference/classes/graph.html#networkx.Graph).
  This allows to ship documentation about the graph's structure to the clients via
  the docstring.
- Interpolation in `Histogram` now rounds the frequency to the nearest integer.

## 🏥 Fixes

- All key/index-specific methods of the database schema now compare candidate keys/indexes
  for an exact match with the given columns. Previously, composite keys/indexes
  would match as soon as they contained the given column.
- Fixed frequency values in most common values and histograms of the `PostgresStatistics`
  catalog.
- Fixed the number of parallel workers of a `QueryPlan` sometimes being wrong if
  the plan originated from PostgreSQL. This is a corner case that was only triggered
  if the optimizer wanted to launch parallel workers but the execution engine was
  unable to do so.
- Restored pretty-printing support for SQL queries in IPython sessions.

## ⚠️ Deprecations

None

## 🪲 Known bugs

- The automatic optimization of the Postgres server configuration as part of the
  Docker installation does not work on MacOS. Currently, this should be considered
  as wontfix.
- The SSB queries can currently not be loaded from the workloads module. The underlying
  data server crashed and we are currently exploring alternative, more reliable
  solutions.

## 🛣 Roadmap

The next months are going to be pretty exciting due to the imminent release of
PostgreSQL 19. As it looks like, this will be the first Postgres version that
provides official support for query hinting via the `pg_plan_advice` extension.
We will definitely add support for pg_plan_advice as a hinting backend, potentially
even making it the standard one and sunsetting development of the `pg_hint_plan`
backend. We will have to wait and see whether pg_hint_plan will be maintained in
the future.

Additionally, the `relalg` module is due for a major overhaul.
