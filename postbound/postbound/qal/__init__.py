"""Contains the basic **query abstraction layer** to conveniently model SQL queries.

The most important features of the qal are:
1. parsing query strings into qal objects
2. providing access to underlying query features such as referenced tables, aliases or predicates
3. formatting qal objects back to strings

Generally, the qal is structured around 3 fundamental concepts: At the core of the qal are SQL expressions (as
specified in the `expressions` module). Such expressions form the elements that are re-used by more high-level
components. For example, there are expressions that model a reference to a column, as well as expressions for function
calls and expressions for modelling math. These expressions are then used to construct predicates (also called
conditions, specified in the `predicates` module) or clauses (specified in the `clauses` module). Predicates are
normally part of clauses (such as a ``WHERE`` clause or a ``HAVING`` clause). Finally, clauses are combined to form the
actual SQL queries (as specified in the `qal` module).

Notice that some references are inherently cyclic: for example, predicates can contain subqueries and
the subqueries in turn contain predicates. This might lead to cyclic import errors in certain corner cases. Such
issues can usually be solved by varying the import sequence slightly.

All concepts in the qal are modelled as data objects that are immutable. In order to modify parts of an SQL query, a
new query has to be constructed. The `transform` module provides some functions to help with that.

In order to generate query instances, the `parser` module can be used to read them from strings. Finally, the
`formatter` module can be used to create pretty representations of queries.

Notes
-----
The immutability enables a very fast hashing of values as well as the caching of complicated computations. Most objects
employ a pattern of determining their hash value during initialization of the object and simply provide that
precomputed value during hashing. This helps to speed up several hot loops at optimization time significantly.
"""
