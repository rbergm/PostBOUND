"""PostBOUND - A research framework for query optimization in relational database systems.

On a high level, PostBOUND is designed for the following workflow: different optimization stratgies - so called optimization
_stages_ - are implemented according to specific interfaces. For example, there are stages that select the join order for an
input query, or stages that compute the cardinality of intermediate results. The different stages are combined in an
`OptimizationPipeline`. Most of the time, the pipeline applies the strategies to each input query in a sequential manner, one
optimization stage after another. The result of the optimization strategies is an abstract description of the optimization
decisions, e.g. which join order to use or how a specific join should be performed. Lastly, the pipeline ensures that the
selected optimization decisions are actually enforced when executing the query and provides an optimized version of the input
query.  Notice that PostBOUND does not interfere with the native optimizer of different database systems directly, since that
would involve a lot of complicated and error-prone code, if it is possible at all. Instead, PostBOUND follows an indirect
approach and makes use of the fact that many database systems (and specifically all of the supported systems) incorporate
proprietary extensions to the SQL standard - mostly in the form of so-called *hint blocks*. These are SQL comments that can be
placed at specific places in the query and have special syntax which tells the optimizer to modify the query execution plan in
a certain way. Alternatively, some systems also interpret certain SQL constructs differently and disable certain optimization
features for them. E.g. by using the explicit *JOIN ON* syntax instead of specifying joins implicitly through *FROM* and
*WHERE* clause, join order optimization can be disabled for some systems. These system-specific properties are utilitized by
PostBOUND to enforce the selected query execution plan at runtime.

In addition to the optimization pipeline, PostBOUND provides a lot of infrastructure to aid in the common tasks of (research
in) query optimization. For example, PostBOUND introduces a high-level query abstraction and provides utilities to parse SQL
queries, apply transformations to them or to access their predicates. Furthermore, a unified interface for different database
systems, e.g. regarding statistics or query plans, allows optimization algorithms to focus on their actual optimization
problem. Likewise, utilities for workloads and benchmarking as well as pre-defined optimization strategies ensure that
evaluations take place on a reproducible foundation.

On a high-level, the PostBOUND project is structured as follows:

- this module contains the actual optimization pipelines and their optimization stages
- the `optimizer` package provides the basic data structures that encode optimization decisions (e.g. selected physical
  operators) as well as general utilities (e.g. a join graph abstraction). Notice that the `optimizer` package is accessible
  under the `opt` alias.
- the `qal` package provides the query abstraction used throughout PostBOUND, as well as logic to parse and transform
  query instances
- the `db` package contains all parts of PostBOUND that concern database interaction. That includes retrieving data
  from different database systems, as well as generating queries for execution based on the optimization decisions
- the `workloads` package provides utilities to load benchmark queries, measure the execution time of different optimization
  pipelines on those benchmarks and offers quick access to some popular workloads
- the `util` package contains algorithms and types that do not belong to specific parts of PostBOUND and are more
  general in nature
- the `vis` package also contains a number of utilities, but with a strict focus on the visualization of different objects that
  are frequently encoutered in the optimization context (such as join trees, query execution plans and join orders). This
  package should only be used if PostBOUND has been installed with visualization support and has to be imported explicitly.

To get a general idea of how to work with PostBOUND and where to start, please take a look at the README and the example
scripts.
Most of the modules are available directly from the main package, so generally you just need to ``import postbound as pb``.
In some cases (e.g. for pre-defined optimization strategies), explicit imports are required. This is described in detail in the
documentation of the respective modules.


Optimization pipeline
---------------------

PostBOUND does not provide a single pipeline implementation. Rather, different pipeline types exists to accomodate
different use-cases. See the documentation of the general `OptimizationPipeline` base class for details. That class serves as
the smallest common denominator among all pipeline implementations. Based on the general interface, the most commonly used
pipelines are the `TextBookOptimizationPipeline` and the `TwoStageOptimizationPipeline`. The former models an optimizer based
on the traditional architecture of cost-based optimizers (i.e. plan enumerator, cost model and cardinality estimator). The
latter first computes a join order and afterwards selects the physical operators for this join order. The resulting plan can be
further parameterized, e.g. using cardinality estimates. Importantly, the `TwoStageOptimizationPipeline` allows to leave some
of the stages empty, which forces the native query optimizer to "fill the gaps" with its own policies. For example, one might
only compute a join order along with the cardinality estimates. The target optimizer would then select the physical operators
based on its own cost model, but using the cardinality estimates in place of its own estimation procedures.

To develop custom optimization algorithms and make use of PostBOUND's pipeline abstraction, the optimization stages are the
interfaces that need to be implemented. They specify the basic interface that pipelines expect and provide additional
information about the selected optimization strategies. Depending on the specific pipeline type, different stages have to be
implemented and each pipeline can require a different amount of steps that need to be applied in a different order. Refer to
the documentation of the respective pipelines for details.


General Workflow
----------------

To implement an optimization strategy, the necessary pipelines as well as its stages need to be identified first. The stages
are designed as abstract interfaces that need to be implemented by the new algorithm. Secondly, a target database has to be
chosen. This is necessary for two reasons: database systems provide different functionality. Therefore, PostBOUND provides some
checks to ensure that the optimization decisions can actually be enforced on the selected database system. Furthermore,
remember that PostBOUND does not actually interfer with the native optimizer of a database system. Instead, it uses optimizer
hints to apply the optimization decisions during query execution. These hints are system-specific and the hint generation
process is also provided by the database system.

An end-to-end optimization scenario typically involves the following steps (some are carried out by PostBOUND automatically,
some require user input):

1. obtaining a working database connection (see the `db` package)
2. setting up the optimization pipeline by configuring the different stages (this is done by the user)
3. building the pipeline
4. loading a workload to optimize (see the `workloads` module)
4. optimizing an input query (this is done by the pipeline)
5. generating the appropriate plan metadata, mostly query hints (this is done by the pipeline and supported by the `db`
   package)
6. executing the input query with the optimization metadata (either manually or using the `executor` module)

"""

from . import (
  db,
  optimizer as opt,
  qal,
  experiments,
  util
)
from ._core import Cost, Cardinality, TableReference
from ._pipelines import (
  OptimizationPipeline,
  IntegratedOptimizationPipeline,
  TextBookOptimizationPipeline,
  TwoStageOptimizationPipeline,
  IncrementalOptimizationPipeline,
  OptimizationSettings
)
from ._stages import (
  CompleteOptimizationAlgorithm,
  CardinalityEstimator, CostModel, PlanEnumerator,
  JoinOrderOptimization, PhysicalOperatorSelection, ParameterGeneration,
  IncrementalOptimizationStep,
  as_complete_algorithm
)
from .db import Database
from .qal import relalg, transform, SqlQuery, ColumnReference, parse_query
from .optimizer import validation, PhysicalQueryPlan
from .experiments import workloads
from .experiments.executor import execute_workload, optimize_and_execute_workload

__version__ = "0.10.0"

__all__ = [
  "db", "opt", "qal", "experiments", "util",
  "OptimizationPipeline",
  "CompleteOptimizationAlgorithm", "IntegratedOptimizationPipeline",
  "Cost", "Cardinality", "TableReference",
  "CardinalityEstimator", "CostModel", "PlanEnumerator", "TextBookOptimizationPipeline",
  "JoinOrderOptimization", "PhysicalOperatorSelection", "ParameterGeneration", "TwoStageOptimizationPipeline",
  "IncrementalOptimizationStep", "IncrementalOptimizationPipeline",
  "as_complete_algorithm", "OptimizationSettings",
  "Database",
  "relalg", "transform", "SqlQuery", "ColumnReference", "parse_query",
  "validation", "PhysicalQueryPlan",
  "workloads", "execute_workload", "optimize_and_execute_workload"
]
