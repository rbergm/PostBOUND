"""Contains PostBOUND's actual query optimization logic as well as the related abstractions and algorithms.

Generally speaking, query optimization in PostBOUND happens in a three-stage process:

1. obtaining an optimized join order for the input query
2. choosing the best-fitting physical operators for the input query and its join order
3. generating additional parameters for the optimized query plan, e.g. parallelization info for individual operators

To accommodate all three stages, PostBOUND introduces an `OptimizationPipeline`. This pipeline can be set up to use
vastly different strategies for each of the stages, depending on the specific research question. Once build, the
pipeline can optimize arbitrary input queries. For the actual query execution, PostBOUND relies on a native database
system instance and does not handle the execution directly. This is because PostBOUND does not interfere with the
optimization process of the database system directly. Instead, it operates as a wrapper around the database and
augments the input query with metadata such that the query plan that was constructed by PostBOUND is enforced at
query execution time. This process can also apply some transformations of the input query.

Each optimization pipeline is created for one specific target database system and database. This is an important
consideration, because the generation of appropriate metadata for the query plan depends on the specific target
database system.

Therefore, an end-to-end optimization scenario involves the following steps (some are carried out by PostBOUND
automatically, some require user input):

1. obtaining a working database connection
2. setting up the optimization pipeline by configuring the different stages
3. building the pipeline
4. optimizing an input query
5. generating the appropriate plan metadata
6. executing the input query with the optimization metadata

To enable an easy adaptation of the optimization pipeline to different research scenarios, all stages of the
optimization process are available behind dedicated interfaces that can be implemented using different strategies.
Notice that PostBOUND chooses empty strategies that simply don't do anything if no dedicated algorithm has been chosen
for a stage.


## Adapting the optimization pipeline

To construct your own optimization pipeline, PostBOUND provides interfaces to customize the behavior of each stage.

The initial join ordering stage determines an optimized join order for the incoming query (defined in package
`joinorder`). How this is achieved, is completely up to the implementation. Furthermore, the algorithm can also
determine an initial selection of physical operators. For example, this can be used to delegate to the native optimizer
of a database system.

In the second stage, the physical operators are determined (defined in package `physops`). Depending on the specific
algorithm and the input data, different scenarios are possible:

- the operator selection may choose all operators according to the specified join order
- the operator selection may choose some operators, leaving the remainder for the native optimizer
- the operator selection may choose operators without a join order, for example if the pipeline is not concerned with
join order optimization at all
- the operator selection may adapt the initial operator selection, thereby overwriting some of the previous assignments
(or throwing them away altogether)

The final stage of the optimization pipeline is concerned with parameterizing the query execution plan further (defined
in package `planmeta`). For example, this can include information about parallel workers for each operator, the
direction of joins or cardinality hints for the native optimizer. Once again, this stage receives the optimized join
order (if there is one) as well as the physical operator assignment (if operators were indeed chosen).

Since each stage is optional, this enables the implementation of vastly different optimization strategies if different
scopes. Some example scenarios are described in the final section.


## Additional infrastructure provided by PostBOUND

Other than interfaces for the different optimization stages, the optimizer package also provides default
implementations and interfaces for some components that are of a more infrastructural nature:

- the `data` module provides a representation of a join tree and a join graph
- the `bounds` package provides interfaces to compute cardinality estimates for (filtered) base tables and
(intermediate) joins. The `stats` module offers an abstract statistics container to automatically store and update
statistics
- the `subqueries` module in the `joinorder` package provides a policy interface to decide, where to include branches
in a join order

To enable a basic introspection of the current optimization settings, all/most interfaces provide a `describe` method.
This method is intended to return a dictionary that contains setting-specific information. Although the format of such
a dict is not standardized (which is why a dict instead of a more rigid object is returned in the first place), all
settings should have a `name` entry which maps to a string that describes the setting's implementation in a
human-readable way. All other contents should be subject to the specific setting. For example, if top-k lists are used
to calculate upper bounds for join estimates, the `JoinEstimator` implementation could provide a dict like the
following: {"name": "topk-estimator", "k": 100}, which indicates that the estimator is based on top-k lists and
processes top-k lists with 100 elements.


## Special scenarios

The flexible combination of the different optimization stages allows to accommodate different optimization scenarios.
Some examples are described in this section.

**Cardinality injection.** Skipping the join ordering and operator selection stages allows to generate cardinalities
for all join candidates in a query during the plan parameterization phase. This way, different cardinality estimators
and their influence on the native query optimizers can be evaluated.

**Pessimistic optimization.** Using an upper bound-driven join ordering algorithm such as UES [0] in combination with
a robust operator selection strategy like TONIC allows to generate and compare different optimization strategies from
the field of defensive or pessimistic query optimization.

**Native optimization.** Delegating the join ordering and operator selection stages to the native optimizers of the
database systems allows to modify specific aspects of their query plans. This can even involve different database
systems. For example, the join order could be obtained from a Postgres instance whereas MySQL handles the operator
selection.
"""
