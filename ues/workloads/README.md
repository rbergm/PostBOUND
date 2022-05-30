# Workload results

This directory contains the incoming queries, as well as the postgres output for different variants of the JOB workload.

## Workloads

| File | Contents |
| ---- | -------- |
| `job-ues-workload-base.csv` | The raw and unmodified UES queries for the JOB workload. These should always be executed with `join_collapse_limit = 1;` and `enable_nestloop = false;` to prevent reordering of any joins and the usage of operators that are unsupported by UES. |
| `job-ues-workload-linearized.csv` | This workload modifies the originial UES queries such that they do not contain any subqueries. Instead, all subqueries are transformed into linear equivalents. Still, the same Postgres parameters apply. |
| `job-ues-workload-idxnlj.csv` | This workload modifies the original UES queries such that the subqueries will be executed using Index Nested Loop joins. The operators are choosen to force the innermost join to be executed as a NestedLoop join and Foreign Key table to be scanned using an Index Scan. The same Postgres parameters as used for the base workload apply. In addition, the hints have to be applied using the _pg\_hint\_plan_ extension.
| `job-ues-workload-idxnlj-allnlj.csv` | This workload modifies the IndexNLJ-workload by forcing the Nested Loop join to be applied to all joins in the subquery. However, the Foreign Key is still the inner relation (i.e. joined based on its index). The same Postgres parameters as used for the base workload apply. |
| `job-ues-workload-idxnlj-allnlj-pkidx.csv` | This workload further modifies the `job-ues-workload-idxnlj-allnlj.csv` queries by also making the Primary Key the inner relation. The same Postgres parameters as used for the base workload apply. |
| `job-ues-workload-idxnlj-pkidx.csv` | This workload modifies the base IndexNLJ-workload by forcing the Primary Key to be the inner relation, but does not restrict the join operators for joins other than the innermost subquery join (which essentially prohibits the use of NLJ for them). The same Postgres parameters as used for the base workload apply. |

