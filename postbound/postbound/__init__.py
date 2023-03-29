"""Main package of the PostBOUND implementation. The `postbound` module contains the optimization pipeline.

On a high-level, the PostBOUND project is structured as follows:

- the `optimizer` package provides the different optimization strategies, interfaces and some pre-defined algorithms
- the `qal` packages provides the query abstraction used throughout PostBOUND, as well as logic to parse and transform
query instances
- the `db` package contains all parts of PostBOUND that concern database interaction. That includes retrieving data
from different database systems, as well as generating optimized queries to execute on the database system
- the `util` package contains algorithms and types that do not belong to specific parts of PostBOUND and are more
general in nature

To get a general idea of how to work with PostBOUND and where to start, take a look at the README.
"""
