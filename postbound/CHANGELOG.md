# Changelog

Updates to PostBOUND are typically made according to the following rules:

1. Version numbers are composed of three numbers, following the typical _major.minor.patch_ scheme. However, until version 1.0
   the boundary between minor versions and patches as well as between major and minor versions is somewhat arbitrary.
2. Minor updates normally should be backwards-compatible and only consist of new functions or parameters with default values.
   The same applies to patches.
3. The version suffixes indicate stability of the current PostBOUND version. No suffix means well-tested and stable, beta means
   decently tested and stable for the most part. Anything below is prototypical.

Notice however that PostBOUND is still a research project. There could be breaking changes if there is very good reason for it.
Be carefull when updating and check the changelog!

---

## Version 0.2.1

### Fixes

- Fixed parsing of `SELECT *` queries for newer versions of _mo\_sql\_parsing_
- Fixed missing visitor implementation for simplified filter views


## Version 0.2.0

This is pretty much a new major version, but we are not ready for 1.0 yet and do not want to convey too much stability.

### New features

- Added a prototypical representation of SQL queries in relational algebra (see `postbound.qal.relalg` module)
- Introduced a Debug mode for the Postgres interface. Enabling it will occasionally print additional internal information.
  Currently, this only includes GeQO settings but it will be expanded in the future.
- When loading workloads column binding can now be enabled explicitly
- Added a `PreComputedCardinalities` service that re-uses existing cardinality estimates
- Database explain nodes now provide a short summary of their subtrees
- Added rudimentary support for predicate equivalence classes. Equivalence classes for equijoins can now be computed and the
  full set of predicates can be generated.
- Created a diff method for join trees that provides human-readable descriptions of the differences between two trees.
- Added an interactive join order builder for linear join orders.
- Join trees, SQL expressions and predicates now support the [Visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern)

### Updates

- The query plans returned by the two-stage optimization pipeline now contain the correct filter and join predicates
- Added `str()` support to plan parameterizations
- `Database` interfaces are now hashable and support equality comparisons
- Improved PostgreSQL setup to support v16 and enable custom ports for the Postgres server
- Various visualization improvements

### Fixes

- Retrieval of existing Postgres instances from the database pool now works as expected
- Postgres query plans for queries without table aliases will no longer duplicate the table name as an alias
- Virtual tables (subquery aliases or `WITH`` clauses) are now bound correctly in their column references
- Retrieval of tables from subqueries now works as expected when calling `SqlQuery.tables()`

---

## Version 0.1.0

### New features

- A utility to generate actual foreign keys for the IMDB has been added to the Postgres tooling. The generation of foreign keys
  can be controlled from the database setup utility.
- A `TimeoutQueryExecutor` has been added for Postgres. It executes a query normally, but raises an error if the query exceeds
  a specific time budget
- A `WorkloadShifter` has been added for Postgres. It enables simple deletes of tuples for arbitrary (Postgres) databases.
- A `CardinalityHintsGenerator` has been added. It eases the implementation of different cardinality estimation strategies
  and the generation of the required metadata.

### Updates

- `pg_buffercache` extension has been added to default Postgres installations and a number of introspection methods for
  internal Postgres state have been added as part of the core utilities for Postgres.
- Query plans can now also contain information about the buffer pool usage
- Accessing methods and attributes on `PostgresExplainPlan` now checks the plan nodes first if the attribute name was not found.
  Afterwards, it still falls back to the normalized plan from the `db` package.
- `PostgresExplainPlan` now supports the _jsonize_ protocol.
- `PostgresExplainPlan` now supports proper hashing and equality checks.
- Improved table prewarming for Postgres: prewarming can now also include indexes
- Improved statistics creation for Postgres: MCV lists and histograms can now be optimized to represent a larger portion of the
  actual data
- Queries from `Workload` instances can be removed using the intersection or subtraction operator by giving an iterable of
  labels

### Fixes

- `transform.replace_clause` now correctly respects subclasses. This also fixes Postgres `LIMIT` clauses not being applied
  correctly while formatting queries for that system.

## Version 0.0.2-beta

- The Postgres interface now tries to be smart about GeQO usage. If a query contains elements that would be overwritten by the
  GeQO optimizer, GeQO is disabled for the current query. Afterwards, the original GeQO configurations is restored. At a later
  point, this behaviour could be augmented to handle all sorts of side effects and restore the original configuration.
- The Postgres `connect` method now re-uses existing (pooled) instances by default. If this is not desired, the `fresh`
  parameter can be set.
