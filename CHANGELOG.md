# Changelog

Updates to PostBOUND are typically made according to the following rules:

1. Version numbers are composed of three numbers, following the typical _major.minor.patch_ scheme. However, until version 1.0
   the boundary between minor versions and patches as well as between major and minor versions is somewhat arbitrary.
2. Minor updates should normally be backwards-compatible and only consist of new functions or parameters with default values.
   The same applies to patches.
3. The version suffixes indicate stability of the current PostBOUND version. No suffix means well-tested and stable, beta means
   decently tested and stable for the most part. Anything below is prototypical.

Notice however that PostBOUND is still a research project. There could be breaking changes if there is very good reason for it.
Be carefull when updating and check the changelog!

---

# ➡ Version 0.13.2

### 🐣 New features
- `format_quick` now supports different SQL flavors (defaulting to standard SQL and also supporting Postgres)

### 💀 Breaking changes
_None_

### 📰 Updates
- Added `__slots__` to all elements of the QAL for improved performance
- The QAL visitors now support optional `*args` and `*kwargs` to supply additional parameters to the visitor

### 🏥 Fixes
- Fixed last couple of parsing/formatting bugs for SQL queries (hopefully)
- Fixed performance regressions when formatting large SQL queries in Postgres-style SQL

### 🪲 Known bugs
- Pre-defined workloads (`workloads.job()`, etc) do not work if installed as a Pip module. This is because the build process
  does not retain the workload directory in the `site_packages`.

### ⏳ WIP
- Baseline for a Postgres-style dynamic programming plan enumerator. This is not yet complete and trying to initialize the
  corresponding class raises an error for now. The enumerator will probably be ready to go for v0.14.0

---


## 🚧 Version 0.14.0 👷

### ⏳ Planned
- Full dynamic programming support for the `TextbookOptimizationPipeline`, including a Postgres-specific algorithm

---

# 🕑 Version 0.13.1

### 🐣 New features
- Added an `expect_match` keyword parameter to `DatabaseSchema.lookup_column`. If this is true, an error is raised if no
  owning table is found (which is the current behavior). Otherwise, _None_ is returned if no match is found.
- Added support for Postgres' *SELECT DISTINCT ON (cols)* syntax. This required a rework of *SELECT* clauses.

### 💀 Breaking changes
- `MathematicalExpression` is now called `MathExpression` for brevity's sake
- Reworked *SELECT* clauses (see above)

### 📰 Updates
- Lifted restriction to support only query plans with 0-2 children. This is necessary to represent _UNION_ statements of more
  than two queries
- `DatabaseSchema.columns` now provides the columns as a sequence instead of a set, ordered by their ordinal position. This is
  necessary to resolve the columns in a *SELECT \** statement correctly.

### 🏥 Fixes
- Lots and lots of fixes for parsing and formatting (very) complicated SQL queries, including correct table resolution
- Fixed some incorrectly named `__match_args__` in the QAL
- Various fixes for `QueryPlan.find_first_node` and `QueryPlan.find_all_nodes`, including checking the subplan if necessary.
- Fixed parsing of *UNION* plans for Postgres

### 🪲 Known bugs
- Pre-defined workloads (`workloads.job()`, etc) do not work if installed as a Pip module. This is because the build process
  does not retain the workload directory in the `site_packages`.
- Parsing workloads and pretty-printing queries now takes considerably longer than in 0.13.0. This is due to the much more
  complicated parsing logic. We will likely do a performance optimization pass in the near future.

### ⏳ WIP
- Baseline for a Postgres-style dynamic programming plan enumerator. This is not yet complete and trying to initialize the
  corresponding class raises an error for now. The enumerator will probably be ready to go for v0.14.0

---


# 🕑 Version 0.13.0

### 🐣 New features
- Table references can now be localized to a schema
- Type casts now support type arguments, e.g. in `varchar(255)`
- Added support for parsing queries with existing hints
- Added support for parsing `EXPLAIN` queries
- Added a very basic dynamic programming enumerator. This is the default enumerator used in the `TextBookOptimizationPipeline`.
  As a WIP, this default should be switched to a Postgres-style DP algorithm if the target database is Postgres. The basic
  enumerator is probably not what you want in production, but better than always forcing the user to supply her own enumerator.
- Added an `indexes_on()` method to the database schema interface. This function returns all indexes for a specific column.

### 💀 Breaking changes
- `virtual` is now a keyword-parameter when creating a table reference
- Removed `MathOperator.Negate` since this clashed with the representation of plain subtraction operations. Instead, negations
  should be represented by `MathOperator.Subtract` and can now be checked with `math_expr.is_unary()`.

### 📰 Updates
- The schema-related Postgres functions now respect the optional table schema (and fall back to _public_ if no schema was
  specified)
- The `NativeCardinalityEstimator` is now an entire `CardinalityHintsGenerator` rather than just a `CardinalityEstimator`
  for more flexible usage.

### 🏥 Fixes
- Fixed a parser error when parsing a unary mathematical expression (e.g. negations)
- Fixed projections in the `SELECT` clause not quoting their alias if necessary
- Fixed text representation of SQL expressions containing a minus
- Fixed `transform.replace_clause()` stopping prematurely which caused some clauses to be dropped unintentionally
- Quoting rules are now also applied to identifiers which contain upper case characters

### 🪲 Known bugs
- Pre-defined workloads (`workloads.job()`, etc) do not work if installed as a Pip module. This is because the build process
  does not retain the workload directory in the `site_packages`.

### ⏳ WIP
- Baseline for a Postgres-style dynamic programming plan enumerator. This is not yet complete and trying to initialize the
  corresponding class raises an error for now. The enumerator will probably be ready to go for v0.14.0

---


# 🕑 Version 0.12.1

### 🐣 New features
- Added a bunch of convenience methods to parts of the QAL, e.g. the `CommonTableExpression` supports `len` now and its CTEs
  can be iterated over directly.

### 💀 Breaking changes
- Renamed the filter predicate when creating function expressions to `filter_where` to align with the property name.

### 📰 Updates
- Added missing `visit_predicate_expr` method to the SQL expression visitor. This was a regression caused by making the
  `AbstractPredicate` an SQL expression.
- Raise a more descriptive error message when parsing a single query of workload fails.

### 🏥 Fixes
- Fixed `transform.rename_table()` only renaming column references to a table, but not the actual table (before, renaming *R*
  to *S* in `SELECT R.a FROM R` produced `SELECT S.a FROM R` rather than `SELECT S.a FROM S`).
- Fixed parsed window functions containing a plain string function name rather than an actual `FunctionExpression`
- Fixed typos in some `__match_args__`
- Fixed `format_quick` not using quoted identifiers in all cases

### 🪲 Known bugs
- Pre-defined workloads (`workloads.job()`, etc) do not work if installed as a Pip module. This is because the build process
  does not retain the workload directory in the `site_packages`.

### ⏳ WIP
- Baseline for dynamic programming plan enumerator. This is not yet complete and trying to initialize a corresponding class
  raises an error for now. The enumerator will probably be ready to go for v0.13.0

---


# 🕑 Version 0.12.0

### 🐣 New features
- Added a new `QueryPlan` class that combines the old `PhysicalQueryPlan` created by the optimizer modules and the old
  `QueryExecutionPlan` created by the database interfaces.
- Added JSON serialization/de-serialization functionality for query plans
- Added support for recursive CTEs

### 💀 Breaking changes
- The `generate_hints` method for databases now uses some named and some default arguments.
- Hash joins are now represented with the hash table as the inner probe side and the outer relation being iterated. This is in
  line with Postgres' implementation.
- Removed the `PhysicalQueryPlan` entirely. Use the unified `QueryPlan` instead
- Removed the `QueryExecutionPlan` entirely. Use the unified `QueryPlan` instead
- Renamed `PostgresExplainPlan.as_query_execution_plan()` to `as_qep()` to be more succinct. The same applies to
  `PostgresExplainNode`.

### 📰 Updates
- Moved the `postgres` module to the top of the package, i.e. you can now do `pb.postgres.connect()`
- Moved the `executor` module to the top of the package, i.e. you can now use `pb.executor.QueryPreparationService`
- Added JSON support to `PhysicalOperatorAssignment` and `PlanParameterization`
- Added a convenience method `add` to the `PhysicalOperatorAssignment`. This method figures out what to add where based on the
  parameters and can be more comfortable to use than `set_scan_operator` and `set_join_operator`
- Added a convenience method `columns_of` to `SqlQuery` to quickly retrieve all columns that belong to a specific query.
- Translating a query into relational algebra now retains ordering information and works for general set queries

### 🏥 Fixes
- Fixed a directory error when creating a SSB database for the first time.

### 🪲 Known bugs
- Pre-defined workloads (`workloads.job()`, etc) do not work if installed as a Pip module. This is because the build process
  does not retain the workload directory in the `site_packages`.

### ⏳ WIP
- Baseline for dynamic programming plan enumerator. This is not yet complete and trying to initialize a corresponding class
  raises an error for now. The enumerator will probably be ready to go for v0.13.0

---


## 🕑 Version 0.11.0

### New features
- Added a `fetch` parameter to `workloads.stack()` which automatically loads the stack queries, if they do not exist
- `SimplifiedFilterView` now tolerates cast expressions since they only modify the data type and not the actual values (and
  PostBOUND does not care about the values anyway).

### Updates
- The `workloads_base_dir` now uses an absolute path based on the location of the workloads module. This should circumvent
  problems with PostBOUND installations as a module.

### Fixes
- Fixed usage of system-specific path separators in `workloads.py` module (looking at you, Windows..)
- Fixed errors being raised during `SimplifiedFilterView.can_wrap` checks

---


## 🕑 Version 0.10.1

### New features
- Added support for set operations to SQL queries
- Added support explicit `VALUES` in SQL queries
- Initializing a `TextbookOptimizationPipeline` without a custom enumerator will now auto-select a dynamic programming-based
  enumerator. If the target database is Postgres, a Postgres-inspired implementation of the algorithm will be used.
- Added `LogicalSqlOperators.Is` and `LogicalSqlOperators.IsNot`
- Added a `is_nullable(<column>)` method to the `DatabaseSchema`
- Added convenience methods `null()` and `is_null()` to `StaticValueExpression` to work with `NULL` values
- Added convenience method `InPredicate.subquery()` to create a new `column IN <subquery>` predicate
- Added optional materialization info to common table expressions
- Added support for lateral subqueries
- Added support for `FILTER` in aggregate functions
- Introduced `__match_args__` for most building blocks of the query abstraction. This allows for much more convenient usage in
  `match` expressions.

### Updates
- Switched to [pglast](https://github.com/lelit/pglast) as the underlying parser for SQL queries. Much better parser
  performance and larger SQL support.
- References to tables and columns now automatically apply quoting if necessary

### ⚠ Breaking changes 💀
- Identities of `TableReference` and `ColumnReference` objects are now based on their lowercase representations. This is a
  necessary change caused by the migration to pglast, since Postgres does not retain the original casing of identifiers after
  parsing.
- Removed `BooleanExpression`. Instead, `AbstractPredicate` is a `SqlExpression` now. This removes the weird distinction
  between predicates and expressions.
- Removed illegal SQL operator "**MISSING**", which was an artifact of the mo-sql parser output
- ``IS NULL`` and ``IS NOT NULL`` predicates are no longer represented by `UnaryPredicate` instances, but by `BinaryPredicate`
  with a *None* static value
- `JoinTableSource` now allows for nested structures. In fact, such sources require both a source as well as a target table
  (which might in turn be arbitrary table sources). This is more in line with the SQL standard and was made possible thanks to
  the transition to the pglast parser.

### WIP
- Baseline for dynamic programming plan enumerator. This is not yet complete and trying to initialize a corresponding class
  raises an error for now. The enumerator will probably be ready to go for v0.11.0

---


## 🕑 Version 0.9.0

**The Docker release**

### New features
- Created a Dockerfile with first-class support. It allows to easily install a local version of PostBOUND complete with a
  ready-to-go Postgres instance.
- Physical query plans now provide information about sorted colums
- Added `--cleanup` switches to all workload setup scripts. These remove the input data files once the database has been
  created.

### Updates
- Added a `.root` property to Postgres query plans to retrieve the root plan node more expressively
- Postgres query plans now also store the plan width of the individual operators (avg. width of tuples produced by the operator)

### WIP
- Baseline for dynamic programming plan enumerator. This is not yet complete and trying to initialize a corresponding class
  raises an error for now. The enumerator will probably be ready to go for v0.10.0

---

## 🕑 Version 0.8.0

### New features
- `tools/setup-py-venv.sh` now provides a one-stop-shop to install PostBOUND as an external package into a Python virtual
  environment
- Added a `timeout` parameter to benchmarking utilities. Notice that timeouts are currently only supported for PostgreSQL.
- Added a (WIP) Dockerfile to create a fresh PostBOUND + Postgres setup

### Updates
- Refactored and re-organized almost the entire code base. For many use-cases it should now be sufficient to only
  `import postbound as pb`. The main module provides the most commonly used types directly. Likewise, double imports such as
  `from postbound.db import db` are no longer necessary.
- The two-stage optimization pipeline no longer tolerates `PhysicalQueryPlan` instances where a `LogicalJoinTree` is expected.
  This clarifies the interfaces a lot and makes development of novel algorithms more straightforward. To ensure a graceful
  handling of older implementations, the two-stage pipeline transforms physical plans into logical join orders as a safeguard.
- Renamed `runner` module to `executor`
- Moved the `bind_columns` function into the parser modules. `transform` no longer depends on the database layer.

### Fixes
- Fixed CTE names being contained twice in the `tables()` output for `SqlQuery` instances
- Fixed auto discovery of the Postgres hinting backend not working on non-POSIX systems

---


## 🕑 Version 0.7.0

### New features
- Added novel `TextBookOptimizationPipeline` for cardinality estimation + cost model + plan enumerator-style algorithms
- Added `AuxiliaryNode`s to physical query plans to represent intermediate computations, e.g. materialization

### Fixes
- Ensure that Postgres interface always updates the GEQO state before running any affected query
- Fixed hinting backend sometimes not initialization internal state correctly

---


## 🕑 Version 0.6.2

### New features
- Process utilities now contain a new `raise_if_error()` method when a command could not be executed
- Database interfaces that support table prewarming can now be queried using the new `PrewarmingSupport` protocol

### Updates
- Reworked detection of hinting backends for the Postgres interface, including much improved error messages. This also allows
  to set the desired hinting backend manually now.
- `jointree.read_from_json()` can now ignore cardinalities
- `Database.describe()` now also contains the global caching mode
- Can now pass the `debug` parameter directly when using `postgres.connect()`
- Lots of fixes and improvements to the database setup utilities

### Fixes
- Fixed out-of-bounds error in `plots.make_grid_plot()` that occurred when all sub-plots could be placed perfectly on the grid
- Fix parsing of queries in `workloads.read_workload()` not respecting the `bind_columns` parameter when loading workloads
  recursively

---


## 🕑 Version 0.6.1

### New features
- Introduced a transformation to automatically generate join equivalence classes

### Updates
- Cardinality estimation policies can now return *None* if they cannot compute an estimate for a specific intermediate. As a
  general rule of thumb, the user should be able to prohibit this behavior when creating a new estimation policy. See
  implementation of the pre-computed cardinalities as an example.

### Fixes
- Make fallback value calculation in pre-computed cardinalities more robust

---


## 🕑 Version 0.6.0

### New features
- Introduced a utility to compute the cardinality of star-queries
- Introduced support for the Stats benchmark
- The PostgreSQL parallel query executor now supports optional timeouts

### Fixes
- Changed the PostgreSQL column lookup to be case-insensitive
- Fix PostgreSQL query hinting to invert the direction of hash joins

---


## 🕑 Version 0.5.0

### New features
- Added support for the [Cardinality Estimation Benchmark](https://www.vldb.org/pvldb/vol14/p2019-negi.pdf)

## 🕑 Version 0.4.6

### New features
- Added a `plots` module to quickly draw plots from a dataframe in grid structure

### Updates
- Improved inspection of subplans in Postgres query plans
- Improved display of subplans in QEP visualization
- Improved rendering of semi join and anti join nodes when visualizing relation algebra plans

### Fixes
- `PostgresExplainNode` now uses more fields to determine its hash value
- Fixed parsing of `NOT IN` predicates in the SQL abstraction
- Fixed parsing of `IN` predicates with subqueries in `relalg` module

---


## 🕑 Version 0.4.5

### Updates
- `PreComputedCardinalities` can now optionally save cardinalities that were computed as part of the live fallback
- `QueryExecutionPlan` nodes now support inputs from subqueries, as is used by e.g. Postgres. These changes are also reflected
  in the `inspect()` output of the plan, as well as in the plan visualization.

### Fixes
- Postgres query plans with more than two child nodes (e.g. for subplans corresponding to subqueries) can now be converted
  correctly to a `QueryExecutionPlan`
- Cardinality hints for Postgres are now always output as integer, never as floats

---


## 🕑 Version 0.4.4

### Updates
- Added a `raw` mode to `execute_query` which does not attempt any simplification of the result set
- PostgreSQL's GeQO optimizer will now be deactivated based on the current hinting backend.

### Fixes
- Removed excessive information from join orders extracted from native query plans in `NativeJoinOrderOptimizer`
- Fixed relalg parsing of `BETWEEN` and `IN` predicates
- Fixed database query cache not storing results as intended

---


## 🕑 Version 0.4.3

### New features
- `PreComputedCardinalities` now support live computation for missing estimates
- Support for [pg_lab](https://github.com/rbergm/pg_lab/) has arrived

### Updates
- SqlQuery now supports _jsonize_ protocol
- Update behaviour to relalg nodes has been reworked
- `EXPLAIN` output now also includes important GUC parameters for Postgres output

## Fixes
- Various fixes to `relalg.Rename` nodes
- Various fixes to cardinality estimation code
- Fixed bugs in query cache
- Broken links for IMDB setup have been fixed

---


## 🕑 Version 0.4.2

### New features
- The `analysis` module now provides a utility to compute the actual plan cost based on the true cardinalities
- The inspection of query plan now accepts a custom lists of fields to be displayed

### Updates
- Warnings from the `db` package now provide categories to enable them to be silenced
- Mutations on `relalg` trees now also update parent nodes
- Added missing method implementations to the rename operator in `relalg`
- The `transform` module now provides a public-API `rename_columns_in_expression`

### Fixes
- Fixed `__str__` implementation for `relalg.Rename`

---


## 🕑 Version 0.4.1

### Updates
- Plan hashes of physical query plans can now be calculated without cardinalities or predicates
- `CustomHashDict` cab now pass additional key-value parameters to the hash function
- Mutations of query plans no longer require mutated child node parameters
- Cleaned up package dependencies


### Fixes
- Fixed bash scripts cancelling prematurely when exit codes in subshells caused `set -e` to abort execution

---


## 🕑 Version 0.4.0

### New features
- Add `mutate` method to all relalg operators to modify their internal state
- Add config script to Postgres management that generates optimized settings for `postgresql.conf` - based on [PGTune by le0pard](https://pgtune.leopard.in.ua/)

### Updates
- `PhysicalQueryPlan` objects can be loaded from query execution plans only based on join order and operators
- After importing the IMDB database into Postgres, the statistics catalog will be updated automatically (including a vacuum)
- When loading a Postgres environment, include paths to the header files will be updated as well
- Complex shell scripts will not exit on first error

### Fixes
- Bugfix for GeQO management in Postgres interface
- Fix Postgres server not stopping correctly after setup (if requested)
- Fix join tree loading from query plans reversing the join directions
- Bugfix for generation of random query plans without duplicates

---


## 🕑 Version 0.3.1

### New features
- Add `Workload.with_labels` method to retrieve sub-workload based on specific labels

### Fixes
- Fixed `PhysicalQueryPlan.load_from_logical_order` creating intermediates of plans instead of tree nodes
- Fixed `PhysicalQueryPlan.plan_hash` calculation for plans with the same logical join structure
- Fixed `ExhaustivePlanEnumerator` scan operator creation
- Fixed `ExhaustiveJoinOrder` not adding join predicates and filters to generated join orders
- Fixed `relalg.parse_relalg` crashing for some predicates involving subqueries

---


## 🕑 Version 0.3.0

### New features
- Added support for Window functions, boolean expressions in SELECT statements and `CASE` expressions
- Added support for nested `JOIN` statements
- Can now check if a table is a database view using the schema

### Updates
- Improved Postgres statistics updates: can now set the actual `n_distinct` value
- Added a `post_repetition_callback` to the workload runner
- During Postgres setup, remote access can now optionally be enabled

---


## 🕑 Version 0.2.1

### Fixes
- Fixed parsing of `SELECT *` queries for newer versions of _mo\_sql\_parsing_
- Fixed missing visitor implementation for simplified filter views

---


## 🕑 Version 0.2.0

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


## 🕑 Version 0.1.0

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

---


## 🕑 Version 0.0.2-beta

- The Postgres interface now tries to be smart about GeQO usage. If a query contains elements that would be overwritten by the
  GeQO optimizer, GeQO is disabled for the current query. Afterwards, the original GeQO configurations is restored. At a later
  point, this behaviour could be augmented to handle all sorts of side effects and restore the original configuration.
- The Postgres `connect` method now re-uses existing (pooled) instances by default. If this is not desired, the `fresh`
  parameter can be set.
