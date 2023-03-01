# Workload results

This directory contains the incoming queries, as well as the postgres output for different variants of the JOB workload.

All queries are executed with `set enable_memoize = 'off';`.

## Basic Workloads

| File | Contents |
| ---- | -------- |
| `job-workload-implicit.` | The original JOB queries, without applying UES. All joins are defined implicitly and the
optimizer can use all of its features. |
| `job-ues-workload-sdr.csv` | The original UES queries as presented in [1]. These should always be executed with
`join_collapse_limit = 1;` and `enable_nestloop = false;` to prevent reordering of any joins and the usage of operators
that are unsupported by UES. |
| `job-ues-workload-base.csv` | The UES queries as generated by the custom UES query generator.  These should always be
executed with `join_collapse_limit = 1;` and `enable_nestloop = false;` to prevent reordering of any joins and the usage
of operators that are unsupported by UES. |

## Top-k workloads

These workloads are located in the `topk-setups` directory and use the improve Top-k join estimation for tighter upper
bounds.