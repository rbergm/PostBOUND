"""Contains PostBOUND's actual query optimization logic as well as the related abstractions and algorithms.

Generally speaking, query optimization in PostBOUND happens in a three-stage process:

1. obtaining an optimized join order for the input query
2. choosing the best-fitting physical operators for the input query and its join order
3. generating additional parameters for the optimized query plan, e.g. parallelization info for individual operators

To accomodate all three stages, PostBOUND introduces an `OptimizationPipeline`. This pipeline can be set up to use
vastly different strategies for each of the stages, depending on the specific research question. Once build, the
pipeline can optimize arbitrary input queries. For the actual query execution, PostBOUND relies on a native database
system instance and does not handle the execution directly. This is because PostBOUND does not interfer with the
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

To enable an easy adaptation of the optimization pipeline to different research scenarious, all stages of the
optimization process are available behind dedicated interfaces that can be implemented using different strategies.
Notice that PostBOUND chooses empty strategies that simply don't do anything if no dedicated algorithm has been chosen
for a stage.

## Adapting the optimization pipeline

TODO
join order
physical operator selection
plan parameterization


## Additional infrastructure provided by PostBOUND

TODO
cardinality estimators
subqueries
stats

To enable a basic introspection of the current optimization settings, all/most interfaces provide a `describe` method.
This method is intended to return a dictionary that contains setting-specific information. Although the format of such
a dict is not standardized (which is why a dict instead of a more rigid object is returned in the first place), all
settings should have a `name` entry which maps to a string that describes the setting's implementation in a
human-readable way. All other contents should be subject to the specific setting. For example, if top-k lists are used
to calculate upper bounds for join estimates, the `JoinEstimator` implementation could provide a dict like the
following: {"name": "topk-estimator", "k": 100}, which indicates that the estimator is based on top-k lists and
processes top-k lists with 100 elements.

## Special scenarios

TODO
scenario cardinality injection
"""
