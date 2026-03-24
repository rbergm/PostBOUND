# Changelog

Version numbers are composed of three components, i.e. _major_._minor_._patch_
As a rough guideline, patch releases are just for fixing bugs or adding minor details (e.g. a new default parameter to
some function), minor releases change slightly larger parts of the framework or add significant new functionality (e.g.
a new optimization pipeline or support for an SQL feature). Major releases fundamentally shift how the framework is used
and indicate stability. Since we are not ready for the 1.0 release yet, this does not matter right now.

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
- Workloads now support transformations of their queries, e.g. `workload.map(pb.transform.as_star_query)`
- Added a `fetch_workload()` function to the workloads module. It allows to load a pre-defined workload by name.
- Added a `n_buffered()` and `buffer_state()` methods to the Postgres statistics interface to retrieve the number of currently
  buffered pages of a relation.
- Added an additional `BoundColumnReference` core type. Instances guarantee to be bound to a `TableReference`.
  The new `assert_bound()` serves as a type guard to narrow references. This should prevent constant checks for valid
  table references on columns.
- Added a `joins_tables()` and `joins_columns()` utilities to query predicates.
- `util.simplify()` now supports single-item mappings as well.
- `util.to_json()` now supports the standard date-like objects (_datetime_, _date_, _time_, and _timedelta_)
- `util.write_df()` now automatically transforms complex objects in the data frame into their JSON representation before
  writing.
- In IPython and Jupyter sessions, common PostBOUND objects like SQL queries or query plans are now automatically
  pretty-printed. Instead of calling `print(plan.explain())`, one can now simply make `plan` the result of a cell.

## 📰 Updates
- `DatabaseStatistics.most_common_values()` now returns an actual `MostCommonValues` object instead of a list of tuples.
  The `MostCommonValues` can be used as a drop-in replacement for the old tuple-based API. In addition, it provides more
  high-level methods for working with the most common values.
- Enabled the MySQL and DuckDB backends to fall back to emulated statistics if the database does not provide them.
- The Postgres `execute_query()` method now accepts hint parameters and automatically applies them. For example, the
  following can now be done without explicit hinting:
```python
query, plan = ...  # whatever
pg_instance.execute_query(query, plan=plan)

# this is equivalent to
hinted_query = pg_instance.hinting().generate_hints(query, plan)
pg_instance.execute_query(hinted_query)
 ```
- The Postgres interface now has a `rollback()` method to put connections back into a valid state.
- The Postgres statistics interface now consistently supports table references with a schema.
- Much improved handling of database schemas during query parsing. We now omit clear warnings in case the database pool is
  is weird.
- The `QueryPreparation` API now provides the `projection` and `output` parameters to modify the *SELECT* clause and the
  type of results to gather for all queries in a more flexible and intuitive way (how did *explain=True*  and
  *analyze=True* interact?).
  The old API using *analyze=True*, etc. is now deprecated in favor of these new parameters.
- Column references now provide a `drop_table_alias()` method to obtain a normalized-ish representation of the column.
  This should be helpful in situations where references to the same column are not consistent in their table references,
  e.g., when one was obtained from the schema and the other was obtained from the query.
- While obtaining a join graph for a query, aliased tables can now be merged into the same node.
- The `PredicateVisitor` can now be started at the query. It will extract the predicates as needed.
- The `to_json()` and `to_json_dump()` utilities now support dataclasses out-of-the-box.
- Expose `argmax()` directly in _util_ module

## 🏥 Fixes
- Fixed `n_buffered()` method of the Postgres statistics interface raising an error if no pages of the relation are currently
  buffered. We now return 0 in this case.
- Fixed string representation of `COUNT(DISTINCT ...)` for multiple arguments. We now generate the correct
  `COUNT(DISTINCT (a, b))` instead of `COUNT(DISTINCT a, b)`.
- Fixed `DatabaseSchema.as_graph()` having the assignment of primary key and foreign key columns reversed on join edges.
- Fixed output format of the benchmarking log if additional entries are appended to an existing log. Essentially, we fix
  such entries being escaped twice.
- Fixed the `standard_logger` sometimes logging internal module names.
- Fixed parser for column JSON

## 💀 Breaking changes
- _None_

## ⚠️ Deprecations
- The old `QueryPrepration` API using *analyze=True*, etc. is now deprecated in favor of the more flexible *projection*
  and *output*  parameters. However, we currently have no plans to remove the old API.

## 🪲 Known bugs
- The automatic optimization of the Postgres server configuration as part of the Docker installation does not work
  on MacOS. Currently, this should be considered as wontfix.

---


# 🛣 Roadmap

Currently, we plan to implement the following features in the future (in no particular order):

- Providing a Substrait export for query plans
- Better benchmarking setup, mostly focused on comparing one or multiple optimization pipelines and creating better
  experiment logs and the ability to cancel/resume long-running benchmarks
