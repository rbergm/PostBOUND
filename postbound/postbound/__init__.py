"""Main package of the PostBOUND implementation.

On a high level, PostBOUND is designed for the following workflow: different optimization stratgies - so called optimization
_stages_ - are implemented according to the interfaces in the `stages` module (this naming will become clear in a second).
Once the specific optimization strategies are instantiated, an `OptimizationPipeline` is configured. This pipeline applies
the strategies for each input query in a sequential manner, one optimization stage after another. The result of the
optimization strategies is an abstract description of the optimization decisions, e.g. which join order to use or how a
specific join should be performed. The final pipeline step is to ensure that the selected optimization decisions are actually
enforced when executing the query. Notice that PostBOUND does not interfere with the native optimizer of different database
systems directly, since that would involve a lot of complicated and error-prone code, if it is possible at all. Instead,
PostBOUND follows an indirect approach and makes use of the fact that many database systems (and specifically all of the
supported systems) incorporate proprietary extensions to the SQL standard - mostly in the form of so-called *hint blocks*.
These are SQL comments that can be placed at specific places in the query and have special content which tells the optimizer to
modify the query execution plan in a certain way. Alternatively, some systems also interpret certain SQL constructs differently
and disable certain optimization features for them. E.g. by using the explicit ``JOIN ON`` syntax instead of specifying joins
implicitly through ``FROM`` and ``WHERE`` clause, join order optimization can be disabled for some systems. These
system-specific properties are utilitized by PostBOUND to enforce the selected query execution plan at runtime.
Therefore, PostBOUND acts as a wrapper around one or several database systems and can access all required information through
well-defined interfaces.

On a high-level, the PostBOUND project is structured as follows:

- the `postbound` module contains the actual optimization pipeline
- the `optimizer` package provides the different optimization strategies, interfaces and some pre-defined algorithms
- the `qal` package provides the query abstraction used throughout PostBOUND, as well as logic to parse and transform
  query instances
- the `db` package contains all parts of PostBOUND that concern database interaction. That includes retrieving data
  from different database systems, as well as generating optimized queries to execute on the database system
- the `workloads` package provides utilities to load benchmark queries, measure the execution time of different optimization
  pipelines on those benchmarks and offers quick access to some popular workloads
- the `util` package contains algorithms and types that do not belong to specific parts of PostBOUND and are more
  general in nature
- the `vis` package also contains a number of utilities, but with a strict focus on the visualization of different objects that
  are frequently encoutered in the optimization context (such as join trees, query execution plans and join orders)

To get a general idea of how to work with PostBOUND and where to start, take a look at the README and the example scripts.
"""

__version__ = "0.6.0"
