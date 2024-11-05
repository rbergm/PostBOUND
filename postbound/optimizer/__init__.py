"""The optimizer package defines the central interfaces to implement optimization algorithms.

In order to optimize input queries, PostBOUND uses so-called *optimization pipelines* that are defined in the `postbound`
module. In contrast, this package is concerned with the building blocks that are used to construct such pipelines, as well as
infrastructure that makes implementing differen optimization algorithms easier.

Generally speaking, the optimization pipelines are structured around different `stages`. Those are defined in the respective
module and can be thought of as individual steps that need to be applied in order to optimize a query. Each pipeline can
require a different amount of steps that need to be applied in a different order. Consult the documentation of the specific
pipelines for more information.

To implement an optimization strategy, the necessary pipelines as well as its stages need to be identified first. The stages
are designed as abstract interfaces that need to be implemented by the new algorithm. Secondly, a target database has to be
chosen. This is necessary for two reasons: database systems provide different functionality. Therefore, PostBOUND provides some
checks to ensure that the optimization decisions can actually be enforced on the selected database system. Furthermore,
remember that PostBOUND does not actually interfer with the native optimizer of a database system. Instead, it uses optimizer
hints to apply the optimization decisions during query execution. These hints are system-specific and the hint generation
process is also provided by the database system.

An end-to-end optimization scenario typically involves the following steps (some are carried out by PostBOUND automatically,
some require user input):

1. obtaining a working database connection
2. setting up the optimization pipeline by configuring the different stages
3. building the pipeline
4. optimizing an input query
5. generating the appropriate plan metadata (mostly query hints)
6. executing the input query with the optimization metadata

In addition to the interfaces for each stage, the optimizer module also provides utility and infrastructure to make the
implementation of optimization algorithms easier. These include:

- a representation for join trees and query execution plans in the `jointree` module
- a representation for join graphs in the `joingraph` module
- interfaces for policies that can be used to parameterize actual optimization algorithms, e.g. to inject different cardinality
  estimation strategies. These are contained in the `policies` package
"""
