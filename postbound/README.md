# PostBOUND

This directory contains the actual (Python) implementation of our PostBOUND framework. The code itself is located in
the `postbound` directory.

## Getting started

```python
from postbound import postbound as pb
from postbound.workloads import workloads
from postbound.optimizer import presets

job = workloads.job()  # let's optimize a query of the Join Order Benchmark
ues = presets.fetch("ues")  # use UES default optimization settings

pipeline = pb.OptimizationPipeline("postgres")  # create our optimization pipeline for Postgres
pipeline.set_join_order_optimization(join_enumerator=ues.join_enumerator,
                                     base_cardinality_estimator=ues.base_cardinality,
                                     join_cardinality_estimator=ues.join_estimator,
                                     subquery_policy=ues.subquery_policy)
pipeline.set_physical_operator_selection(operator_selector=ues.operator_selection)
pipeline.build()  # finalize the pipeline to be ready for optimization

pipeline.optimize_query(job["1a"])
```
