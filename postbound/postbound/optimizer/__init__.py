"""Contains abstractions and algorithms for the actual join order optimization and operator selection.

More specifically the following basic interfaces are provided:

TODO
join order
physical operator selection

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

TODO
scenario cardinality injection
"""
