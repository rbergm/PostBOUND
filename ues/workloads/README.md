# Workload results

| File | Contents |
| ---- | -------- |
| `job-ues-results-fks-nonlj.csv` | The main results file. Contains results of both the original UES workload, as well as the transformed (i.e. subquery-removed) workload. The workloads were executed with Nested Loop joins disabled (i.e. `set enable_nestloop = false;`), join reordering disabled (i.e. `set join_collapse_limit = 1;`) and all foreign key/secondary indices present.

