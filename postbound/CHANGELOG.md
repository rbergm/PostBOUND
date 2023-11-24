# Changelog

Updates to PostBOUND are typically made according to the following rules:

1. Version numbers are composed of three numbers, following the typical _major.minor.patch_ scheme. However, until version 1.0
   the boundary between minor versions and patches is somewhat arbitrary
2. Minor updates normally should be backwards-compatible and only consist of new functions or parameters with default values.
   The same applies to patches.
3. The version suffixes indicate stability of the current PostBOUND version. No suffix means well-tested and stable, beta means
   decently tested and stable for the most part. Anything below is prototypical.

Notice however that PostBOUND is still a research project. There could be breaking changes if there is very good reason for it.
Be carefull when updating and check the changelog!

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
