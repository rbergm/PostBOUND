"""Contains the basic **query abstraction layer** to conveniently operate on SQL queries on a logical level.

The most important features of the qal are:
1. parsing query strings into qal objects
2. providing access to underlying query features such as referenced tables, aliases or predicates
3. formatting qal objects back to strings

Generally, the qal is structured around 3 fundamental concepts: At the core of the qal are SQL expressions (as
specified in the `expressions` module). Such expressions form the elements that are re-used by more high-level
components. For example, a column reference can be an expression, as well as a function over columns or mathematical
expressions. These expressions are then used to construct predicates (also called conditions, specified in the
`expressions` module) or clauses (specified in the `clauses` module). Predicates are normally part of clauses (such as
a WHERE clause or a HAVING clause). Finally, clauses are combined to form the actual SQL queries (as specified in the
`qal` module).

Notice that some references are inherently cyclic: for example, predicates can contain subqueries and
the subqueries in turn contain predicates. This might lead to cyclic import errors in certain corner cases. Such
issues can usually be solved by varying the import sequence slightly.

The `parser`, `formatter` and `transform` modules operate on top of qal objects or handle their construction.
"""
