"""Contains the basic **query abstraction layer** to conveniently model SQL queries.

The most important features of the qal are:
1. parsing query strings into qal objects
2. providing access to underlying query features such as referenced tables, aliases or predicates
3. converting queries to representations in relational algebra
4. formatting qal objects back to strings

Generally, the qal is structured around 3 fundamental concepts: At the core of the qal are SQL expressions. Such expressions
form the basic building blocks that are re-used by more high-level components. For example, there are expressions that model a
reference to a column, as well as expressions for function calls and expressions for modelling math. The `SqlExpression` acts
as the common base class for all different expression types.

Expressions are used to construct predicates or clauses. Predicates are normally part of clauses (such as a *WHERE* clause or
a *HAVING* clause). Finally, clauses are combined to form the actual SQL queries.

Using these basic building blocks, the `relalg` module provides a simple model of relational algebra, as well as means to
translate a parsed SQL query to an algebraic expression.

A common pattern when working with elements of the qal are the `tables` and `columns` methods (along with some other, more
rarely used ones). These are defined on pretty much all of the qal types and provide access to the tables, respectively the
columns that are referenced within the current element.

Notice that some references in the qal are inherently cyclic: for example, predicates can contain subqueries and
the subqueries in turn contain predicates. This might lead to cyclic import errors in certain corner cases. Such
issues can usually be solved by varying the import sequence slightly.

All concepts in the qal are modelled as data objects that are immutable. In order to modify parts of an SQL query, a
new query has to be constructed. The `transform` module provides some functions to help with that. Traversal of the different
parts of a query can be done using specific visitor implementations.

In order to generate query instances, the `parser` module can be used to read them from strings. Finally, the
`formatter` module can be used to create pretty representations of queries. The `transform`, and `formatter` modules are
available directly from the qal and do not need to be imported explicitly. The same holds for a simple relational algebra
representation in the `relalg` module. The parser provides means for reading an entire query from text, or reading parts of it
from JSON. A `parse_query` helper function is directly available from the qal module.


SQL queries
-----------

The most important type of our query abstraction is the `SqlQuery` class. It focuses on modelling an entire SQL query with all
important concepts. Notice that the focus here really in on modelling - nearly no interactive functionality, no input/output
capabilities and no modification tools are provided. These are handled by dedicated modules (e.g. the `parser` module for
reading queries from text, or the `transform` module for changing existing query objects).

In addition to the pure `SqlQuery`, a number of subclasses exist. These model queries with specific *FROM* clauses. For
example, the `ImplicitSqlQuery` provides an `ImplicitFromClause` that restricts how tables can be referenced in this clause.
For some use-cases, these might be easier to work with than the more general `SqlQuery` class, where much more diverse *FROM*
clauses are permitted.


Predicates
----------

Predicates are the central building block to represent filter conditions for SQL queries.

A predicate is a boolean expression that can be applied to a tuple to determine whether it should be kept in the intermediate
result or thrown away. PostBOUND distinguishes between two kinds of predicates, even though they are both represented by the
same class: there are filter predicates, which - as a rule of thumb - can be applied directly to base table relations.
Furthermore, there are join predicates that access tuples from different relations and determine whether the join of both
tuples should become part of the intermediate result.

PostBOUND's implementation of predicates is structured using a composite-style layout: The `AbstractPredicate` interface
describes all behaviour that is common to the concrete predicate types. There are `BasePredicate`s, which typically contain
different expressions. The `CompoundPredicate` is used to nest different predicates, thereby creating tree-shaped hierarchies.

In addition to the predicate representation, this module also provides a utility for streamlined access to the important parts
of simple filter predicates via the `SimplifiedFilterView`.
Likwise, the `QueryPredicates` provide high-level access to all predicates (join and filter) that are specified in a query.
From a user perspective, this is probably the best entry point to work with predicates. Alternatively, the predicate tree can
also be traversed using custom functions.

Lastly, there exists some basic support for equivalence class computation via the `determine_join_equivalence_classes` and
`generate_predicates_for_equivalence_classes` functions.


Clauses
-------

In addition to widely accepted clauses such as the default SPJ-building blocks or grouping clauses (*GROUP BY* and
*HAVING*), some additional clauses are also defined. These include `Explain` clauses that model widely
used *EXPLAIN* queries which provide the query plan instead of optimizing the query. Furthermore, the `Hint` clause
is used to model hint blocks that can be used to pass additional non-standardized information to the database system
and its query optimizer. In real-world contexts this is mostly used to correct mistakes by the optimizer, but PostBOUND
uses this feature to enforce entire query plans. The specific contents of a hint block are not standardized by
PostBOUND and thus remains completely system-specific.

All clauses inherit from `BaseClause`, which specifies the basic common behaviour shared by all concrete clauses.
Furthermore, all clauses are designed as immutable data objects whose content cannot be changed. Any forced
modifications will break the entire query abstraction layer and lead to unpredictable behaviour.


Notes
-----
The immutability enables a very fast hashing of values as well as the caching of complicated computations. Most objects
employ a pattern of determining their hash value during initialization of the object and simply provide that
precomputed value during hashing. This helps to speed up several hot loops at optimization time significantly.
"""

from __future__ import annotations

from typing import Optional

from . import relalg, transform
from ._qal import (
    MathOperator,
    LogicalOperator,
    UnarySqlOperators,
    CompoundOperator,
    SqlOperator,
    SqlExpression,
    StaticValueExpression,
    CastExpression,
    MathExpression,
    ColumnExpression,
    AggregateFunctions,
    FunctionExpression,
    ArrayAccessExpression,
    SubqueryExpression,
    StarExpression,
    WindowExpression,
    CaseExpression,
    SqlExpressionVisitor,
    ExpressionCollector,
    as_expression,
    NoJoinPredicateError,
    NoFilterPredicateError,
    BaseExpression,
    AbstractPredicate,
    BasePredicate,
    BinaryPredicate,
    BetweenPredicate,
    InPredicate,
    UnaryPredicate,
    CompoundPredicate,
    PredicateVisitor,
    as_predicate,
    determine_join_equivalence_classes,
    generate_predicates_for_equivalence_classes,
    UnwrappedFilter,
    SimplifiedFilterView,
    QueryPredicates,
    DefaultPredicateHandler,
    BaseClause,
    Hint,
    Explain,
    WithQuery,
    CommonTableExpression,
    BaseProjection,
    DistinctType,
    Select,
    TableSource,
    DirectTableSource,
    SubqueryTableSource,
    ValuesTableSource,
    JoinType,
    JoinTableSource,
    From,
    ImplicitFromClause,
    ExplicitFromClause,
    Where,
    GroupBy,
    Having,
    OrderByExpression,
    OrderBy,
    Limit,
    UnionClause,
    IntersectClause,
    ExceptClause,
    SetOperationClause,
    ClauseVisitor,
    collect_subqueries_in_expression,
    FromClauseType,
    SqlQuery,
    ImplicitSqlQuery,
    ExplicitSqlQuery,
    MixedSqlQuery,
    SetQuery,
    SelectStatement,
    SqlStatement,
    build_query
)
from .parser import DBCatalog
from .formatter import format_quick
from .._core import TableReference, ColumnReference, UnboundColumnError, VirtualTableError, quote, normalize


__all__ = [
    "MathOperator",
    "LogicalOperator",
    "UnarySqlOperators",
    "CompoundOperator",
    "SqlOperator",
    "SqlExpression",
    "StaticValueExpression",
    "CastExpression",
    "MathExpression",
    "ColumnExpression",
    "AggregateFunctions",
    "FunctionExpression",
    "ArrayAccessExpression",
    "SubqueryExpression",
    "StarExpression",
    "WindowExpression",
    "CaseExpression",
    "SqlExpressionVisitor",
    "ExpressionCollector",
    "as_expression",
    "NoJoinPredicateError",
    "NoFilterPredicateError",
    "BaseExpression",
    "AbstractPredicate",
    "BasePredicate",
    "BinaryPredicate",
    "BetweenPredicate",
    "InPredicate",
    "UnaryPredicate",
    "CompoundPredicate",
    "PredicateVisitor",
    "as_predicate",
    "determine_join_equivalence_classes",
    "generate_predicates_for_equivalence_classes",
    "UnwrappedFilter",
    "SimplifiedFilterView",
    "QueryPredicates",
    "DefaultPredicateHandler",
    "BaseClause",
    "Hint",
    "Explain",
    "WithQuery",
    "CommonTableExpression",
    "BaseProjection",
    "DistinctType",
    "Select",
    "TableSource",
    "DirectTableSource",
    "SubqueryTableSource",
    "ValuesTableSource",
    "JoinType",
    "JoinTableSource",
    "From",
    "ImplicitFromClause",
    "ExplicitFromClause",
    "Where",
    "GroupBy",
    "Having",
    "OrderByExpression",
    "OrderBy",
    "Limit",
    "UnionClause",
    "IntersectClause",
    "ExceptClause",
    "ClauseVisitor",
    "SetOperationClause",
    "collect_subqueries_in_expression",
    "FromClauseType",
    "SqlQuery",
    "ImplicitSqlQuery",
    "ExplicitSqlQuery",
    "MixedSqlQuery",
    "SetQuery",
    "SelectStatement",
    "SqlStatement",
    "build_query",
    "relalg",
    "transform",
    "format_quick",
    "TableReference",
    "ColumnReference",
    "UnboundColumnError",
    "VirtualTableError",
    "quote",
    "normalize",
    "parse_query",
    "parse_full_query"
]


def parse_query(query: str, *, include_hints: bool = True,
                bind_columns: Optional[bool] = None,
                db_schema: Optional[DBCatalog] = None) -> SqlQuery:
    """Parses a query string into a proper `SqlQuery` object.

    During parsing, the appropriate type of SQL query (i.e. with implicit, explicit or mixed *FROM* clause) will be
    inferred automatically. Therefore, this method can potentially return a subclass of `SqlQuery`.

    Once the query has been transformed, a text-based binding process is executed. During this process, the referenced
    tables are normalized such that column references using the table alias are linked to the correct tables that are
    specified in the *FROM* clause (see the module-level documentation for an example). The parsing process can
    optionally also involve a binding process based on the schema of a live database. This is important for all
    remaining columns where the text-based parsing was not possible, e.g. because the column was specified without a
    table alias.

    Parameters
    ----------
    query : str
        The query to parse
    include_hints : bool, optional
        Whether to include hints in the parsed query. If this is *True* (the default), any preceding comments in the query
        text will be parsed as a hint block. Otherwise, these comments are simply ignored.
    bind_columns : bool | None, optional
        Whether to use *live binding*. This does not control the text-based binding, which is always performed. If this
        parameter is *None* (the default), the global `auto_bind_columns` variable will be queried. Depending on its
        value, live binding will be performed or not.
    db_schema : Optional[DBCatalog], optional
        For live binding, this indicates the database to use. If this is *None* (the default), the database will be
        tried to extract from the `DatabasePool`

    Returns
    -------
    SqlQuery
        The parsed SQL query.
    """
    from .parser import parse_query as parse_worker
    return parse_worker(query, accept_set_query=False, include_hints=include_hints,
                        bind_columns=bind_columns, db_schema=db_schema)


def parse_full_query(statement: str, *, bind_columns: Optional[bool] = None,
                     db_schema: Optional[DBCatalog] = None) -> SelectStatement:
    """This method is very similar to `parse_query`, but it also support set queries (i.e. queries with **UNION**, etc.).

    See Also
    --------
    parse_query : The simpler version of this method that only supports "plain" queries without set operations.
    """
    from .parser import parse_query as parse_worker
    return parse_worker(statement, accept_set_query=True, include_hints=True,
                        bind_columns=bind_columns, db_schema=db_schema)
