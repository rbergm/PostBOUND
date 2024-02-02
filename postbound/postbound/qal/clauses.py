"""Models clauses of SQL queries, like ``SELECT``, ``FROM`` or ``WITH``.

In addition to widely accepted clauses such as the default SPJ-building blocks or grouping clauses (``GROUP BY`` and
``HAVING``), some additional clauses are also defined here. These include the `Explain` clauses that models the widely
used ``EXPLAIN`` queries that provide the query plan instead of optimizing the query. Furthermore, the `Hint` clause
is used to model hint blocks that can be used to pass additional non-standardized information to the database system
and its query optimizer. In real-world contexts this is mostly used to correct mistakes by the optimizer, but PostBOUND
uses this feature to enforce entire query plans. The specific contents of a hint block are not standardized at all by
PostBOUND and thus remains completely system-specific.

All clauses inherit from `BaseClause`, which specifies the basic common behaviour shared by all concrete clauses.
Furthermore, all clauses are designed as immutable data objects whose content cannot be changed. Any forced
modifications will break the entire query abstraction layer and lead to unpredictable behaviour.
"""
from __future__ import annotations

import abc
import enum
import typing
from collections.abc import Iterable, Sequence
from typing import Optional
from postbound.qal import base, expressions as expr, qal, predicates as preds
from postbound.util import collections as collection_utils


# TODO: make handling of optional string arguments/properites consistent (empty string vs None)


class BaseClause(abc.ABC):
    """Basic interface shared by all supported clauses.

    This really is an abstract interface, not a usable clause. All inheriting clauses have to provide their own
    `__eq__` method and re-use the `__hash__` method provided by the base clause. Remember to explicitly set this up!
    The concrete hash value is constant since the clause itself is immutable. It is up to the implementing class to
    make sure that the equality/hash consistency is enforced.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete clause object
    """

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced in the clause.

        Returns
        -------
        set[base.TableReference]
            All tables. This includes virtual tables if such tables are present in the clause
        """
        return collection_utils.set_union(expression.tables() for expression in self.iterexpressions())

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced in the clause.

        Returns
        -------
        set[base.ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all directly contained expressions in this clause.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[expr.SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides access to all column in this clause.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n*
        times in this clause, it will also be returned *n* times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.

        Returns
        -------
        Iterable[base.ColumnReference]
            All columns exactly in the order in which they are used
        """
        raise NotImplementedError

    @abc.abstractmethod
    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        """Enables processing of the current clause by a visitor.

        Parameters
        ----------
        visitor : ClauseVisitor
            The visitor
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class Hint(BaseClause):
    """Hint block of a clause.

    Depending on the SQL dialect, these hints will be placed at different points in the query. Furthermore, the precise
    contents (i.e. syntax and semantic) vary from database system to system.

    Hints are differentiated in two parts:

    - preparatory statements can be executed as valid commands on the database system, e.g. optimizer settings, etc.
    - query hints are the actual hints. Typically, these will be inserted as comments at some place in the query.

    These two parts are set as parameters in the `__init__` method and are available as read-only properties
    afterwards.

    Parameters
    ----------
    preparatory_statements : str, optional
        Statements that configure the optimizer and have to be run *before* the actual query is executed. Such settings
        often configure the optimizer for the entire session and can thus influence other queries as well. Defaults to
        an empty string, which indicates that there are no such settings.
    query_hints : str, optional
        Hints that configure the optimizer, often for an individual join. These hints are executed as part of the
        actual query.

    Examples
    --------
    A hint clause for MySQL could look like this:

    .. code-block::sql

        SET optimizer_switch = 'block_nested_loop=off';
        SELECT /*+ HASH_JOIN(R S) */ R.a
        FROM R, S, T
        WHERE R.a = S.b AND S.b = T.c

    This enforces the join between tables *R* and *S* to be executed as a hash join (due to the query hint) and
    disables usage of the block nested-loop join for the entire query (which in this case only affects the join between
    tables *S* and *T*) due to the preparatory ``SET optimizer_switch`` statement.
    """

    def __init__(self, preparatory_statements: str = "", query_hints: str = ""):
        self._preparatory_statements = preparatory_statements
        self._query_hints = query_hints

        hash_val = hash((preparatory_statements, query_hints))
        super().__init__(hash_val)

    @property
    def preparatory_statements(self) -> str:
        """Get the string of preparatory statements. Can be empty.

        Returns
        -------
        str
            The preparatory statements. If these are multiple statements, they are concatenated into a single string
            with appropriate separator characters between them.
        """
        return self._preparatory_statements

    @property
    def query_hints(self) -> str:
        """Get the query hint text. Can be empty.

        Returns
        -------
        str
            The hints. The string has to be understood as-is by the target database system. If multiple hints are used,
            they have to be concatenated into a single string with appropriate separator characters between them.
            Correspondingly, if the hint blocks requires a specific prefix/suffix (e.g. comment syntax), this has to
            be part of the string as well.
        """
        return self._query_hints

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_hint_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.preparatory_statements == other.preparatory_statements
                and self.query_hints == other.query_hints)

    def __str__(self) -> str:
        if self.preparatory_statements and self.query_hints:
            return self.preparatory_statements + "\n" + self.query_hints
        elif self.preparatory_statements:
            return self.preparatory_statements
        return self.query_hints


class Explain(BaseClause):
    """``EXPLAIN`` block of a query.

    ``EXPLAIN`` queries change the execution mode of a query. Instead of focusing on the actual query result, an
    ``EXPLAIN`` query produces information about the internal processes of the database system. Typically, this
    includes which execution plan the DBS would choose for the query. Additionally, ``EXPLAIN ANALYZE`` (as for example
    supported by Postgres) provides the query plan and executes the actual query. The returned plan is then annotated
    by how the optimizer predictions match reality. Furthermore, such ``ANALYZE`` plans typically also contain some
    runtime statistics such as runtime of certain operators.

    Notice that there is no ``EXPLAIN`` keyword in the SQL standard, but all major database systems provide this
    functionality. Nevertheless, the precise syntax and semantic of an ``EXPLAIN`` statement depends on the actual DBS.
    The Explain clause object is modeled after Postgres and needs to adapted accordingly for different systems (see
    `db.HintService`). Especially the ``EXPLAIN ANALYZE`` variant is not supported by all systems.

    Parameters
    ----------
    analyze : bool, optional
        Whether the query should not only be executed as an ``EXPLAIN`` query, but rather as an ``EXPLAIN ANALYZE``
        query. Defaults to ``False`` which runs the query as a pure ``EXPLAIN`` query.
    target_format : Optional[str], optional
        The desired output format of the query plan, if this is supported by the database system. Defaults to ``None``
        which normally forces the default output format.

    See Also
    --------
    postbound.db.db.HintService.format_query

    References
    ----------

    .. PostgreSQL ``EXPLAIN`` command: https://www.postgresql.org/docs/current/sql-explain.html
    """

    @staticmethod
    def explain_analyze(format_type: str = "JSON") -> Explain:
        """Constructs an ``EXPLAIN ANALYZE`` clause with the specified output format.

        Parameters
        ----------
        format_type : str, optional
            The output format, by default ``"JSON"``

        Returns
        -------
        Explain
            The explain clause
        """
        return Explain(True, format_type)

    @staticmethod
    def plan(format_type: str = "JSON") -> Explain:
        """Constructs a pure ``EXPLAIN`` clause (i.e. without ``ANALYZE``) with the specified output format.

        Parameters
        ----------
        format_type : str, optional
            The output format, by default ``"JSON"``

        Returns
        -------
        Explain
            The explain clause
        """
        return Explain(False, format_type)

    def __init__(self, analyze: bool = False, target_format: Optional[str] = None):
        self._analyze = analyze
        self._target_format = target_format if target_format != "" else None

        hash_val = hash((analyze, target_format))
        super().__init__(hash_val)

    @property
    def analyze(self) -> bool:
        """Check, whether the query should be executed as ``EXPLAIN ANALYZE`` rather than just plain ``EXPLAIN``.

        Usually, ``EXPLAIN ANALYZE`` executes the query and gathers extensive runtime statistics (e.g. comparing
        estimated vs. true cardinalities for intermediate nodes).

        Returns
        -------
        bool
            Whether ``ANALYZE`` mode is enabled
        """
        return self._analyze

    @property
    def target_format(self) -> Optional[str]:
        """Get the target format in which the ``EXPLAIN`` plan should be provided.

        Returns
        -------
        Optional[str]
            The output format, or ``None`` if this is not specified. This is never an empty string.
        """
        return self._target_format

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_explain_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.analyze == other.analyze
                and self.target_format == other.target_format)

    def __str__(self) -> str:
        explain_prefix = "EXPLAIN"
        explain_body = ""
        if self.analyze and self.target_format:
            explain_body = f" (ANALYZE, FORMAT {self.target_format})"
        elif self.analyze:
            explain_body = " ANALYZE"
        elif self.target_format:
            explain_body = f" (FORMAT {self.target_format})"
        return explain_prefix + explain_body


class WithQuery:
    """A single common table expression that can be referenced in the actual query.

    Each ``WITH`` clause can consist of multiple auxiliary common table expressions. This class models exactly one
    such query. It consists of the query as well as the name under which the temporary table can be referenced
    in the actual SQL query.

    Parameters
    ----------
    query : qal.SqlQuery
        The query that should be used to construct the temporary common table.
    target_name : str
        The name under which the table should be made available

    Raises
    ------
    ValueError
        If the `target_name` is empty
    """
    def __init__(self, query: qal.SqlQuery, target_name: str) -> None:
        if not target_name:
            raise ValueError("Target name is required")
        self._query = query
        self._subquery_expression = expr.SubqueryExpression(query)
        self._target_name = target_name
        self._hash_val = hash((query, target_name))

    @property
    def query(self) -> qal.SqlQuery:
        """The query that is used to construct the temporary table

        Returns
        -------
        qal.SqlQuery
            The query
        """
        return self._query

    @property
    def subquery(self) -> expr.SubqueryExpression:
        """Provides the query that constructsd the temporary table as a subquery object.

        Returns
        -------
        expr.SubqueryExpression
            The subquery
        """
        return self._subquery_expression

    @property
    def target_name(self) -> str:
        """The table name under which the temporary table can be referenced in the actual SQL query

        Returns
        -------
        str
            The name. Will never be empty.
        """
        return self._target_name

    @property
    def target_table(self) -> base.TableReference:
        """The table under which the temporary CTE table can be referenced in the actual SQL query

        The only difference to `target_name` is the type of this property: it provides a proper (virtual) table
        reference object

        Returns
        -------
        base.TableReference
            The table. Will always be a virtual table.
        """
        return base.TableReference.create_virtual(self.target_name)

    def tables(self) -> set[base.TableReference]:
        return self._query.tables()

    def columns(self) -> set[base.ColumnReference]:
        return self._query.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [self._subquery_expression]

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self._query.itercolumns()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._target_name == other._target_name
                and self._query == other._query)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        query_str = self._query.stringify(trailing_delimiter=False)
        return f"{self._target_name} AS ({query_str})"


class CommonTableExpression(BaseClause):
    """The ``WITH`` clause of a query, consisting of at least one CTE query.

    Parameters
    ----------
    with_queries : Iterable[WithQuery]
        The common table expressions that form the WITH clause.

    Raises
    ------
    ValueError
        If `with_queries` does not contain any CTE

    """
    def __init__(self, with_queries: Iterable[WithQuery]):
        self._with_queries = tuple(with_queries)
        if not self._with_queries:
            raise ValueError("With queries cannnot be empty")
        super().__init__(hash(self._with_queries))

    @property
    def queries(self) -> Sequence[WithQuery]:
        """Get CTEs that form the ``WITH`` clause

        Returns
        -------
        Sequence[WithQuery]
            The CTEs in the order in which they were originally specified.
        """
        return self._with_queries

    def tables(self) -> set[base.TableReference]:
        all_tables: set[base.TableReference] = set()
        for cte in self._with_queries:
            all_tables |= cte.tables() | {cte.target_table}
        return all_tables

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(with_query.columns() for with_query in self._with_queries)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten(with_query.iterexpressions() for with_query in self._with_queries)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(with_query.itercolumns() for with_query in self._with_queries)

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_cte_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._with_queries == other._with_queries

    def __str__(self) -> str:
        query_str = ", ".join(str(with_query) for with_query in self._with_queries)
        return "WITH " + query_str


class BaseProjection:
    """The `BaseProjection` forms the fundamental building block of a ``SELECT`` clause.

    Each ``SELECT`` clause is composed of at least one base projection. Each projection can be an arbitrary
    `SqlExpression` (rules and restrictions of the SQL standard are not enforced here). In addition, each projection
    can receive a target name as in ``SELECT foo AS f FROM bar``.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression that is used to calculate the column value. In the simplest case, this can just be a
        `ColumnExpression`, which provides the column values directly.
    target_name : str, optional
        An optional name under which the column should be accessible. Defaults to an empty string, which indicates that
        the original column value or a system-specific modification of that value should be used. The latter case
        mostly applies to columns which are modified in some way, e.g. by a mathematical expression or a function call.
        Depending on the specific database system, the default column name could just be the function name, or the
        function name along with all its parameters.

    """

    @staticmethod
    def count_star() -> BaseProjection:
        """Shortcut method to create a ``COUNT(*)`` projection.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(expr.FunctionExpression("count", [expr.StarExpression()]))

    @staticmethod
    def star() -> BaseProjection:
        """Shortcut method to create a ``*`` (as in ``SELECT * FROM R``) projection.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(expr.StarExpression())

    @staticmethod
    def column(col: base.ColumnReference, target_name: str = "") -> BaseProjection:
        """Shortcut method to create a projection for a specific column.

        Parameters
        ----------
        col : base.ColumnReference
            The column that should be projected
        target_name : str, optional
            An optional name under which the column should be available.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(expr.ColumnExpression(col), target_name)

    def __init__(self, expression: expr.SqlExpression, target_name: str = ""):
        if not expression:
            raise ValueError("Expression must be set")
        self._expression = expression
        self._target_name = target_name
        self._hash_val = hash((expression, target_name))

    @property
    def expression(self) -> expr.SqlExpression:
        """Get the expression that forms the column.

        Returns
        -------
        expr.SqlExpression
            The expression
        """
        return self._expression

    @property
    def target_name(self) -> str:
        """Get the alias under which the column should be accessible.

        Can be empty to indicate the absence of a target name.

        Returns
        -------
        str
            The name
        """
        return self._target_name

    def columns(self) -> set[base.ColumnReference]:
        return self.expression.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.expression.itercolumns()

    def tables(self) -> set[base.TableReference]:
        return self.expression.tables()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.expression == other.expression and self.target_name == other.target_name)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self.target_name:
            return str(self.expression)
        return f"{self.expression} AS {self.target_name}"


class SelectType(enum.Enum):
    """Indicates the specific type of the ``SELECT`` clause."""

    Select = "SELECT"
    """Plain projection without duplicate removal."""

    SelectDistinct = "SELECT DISTINCT"
    """Projection with duplicate elimination."""


class Select(BaseClause):
    """The ``SELECT`` clause of a query.

    This is the only required part of a query. Everything else is optional and can be left out. (Notice that PostBOUND
    is focused on SPJ-queries, hence there are no ``INSERT``, ``UPDATE``, or ``DELETE`` queries)

    A ``SELECT`` clause simply consists of a number of individual projections (see `BaseProjection`), its `targets`.

    Parameters
    ----------
    targets : BaseProjection | Sequence[BaseProjection]
        The individual projection(s) that form the ``SELECT`` clause
    projection_type : SelectType, optional
        The kind of projection that should be performed (i.e. with duplicate elimination or without). Defaults
        to a `SelectType.Select`, which is a plain projection without duplicate removal.

    Raises
    ------
    ValueError
        If the `targets` are empty.
    """

    @staticmethod
    def count_star() -> Select:
        """Shortcut method to create a ``SELECT COUNT(*)`` clause.

        Returns
        -------
        Select
            The clause
        """
        return Select(BaseProjection.count_star())

    @staticmethod
    def star() -> Select:
        """Shortcut to create a ``SELECT *`` clause.

        Returns
        -------
        Select
            The clause
        """
        return Select(BaseProjection.star())

    @staticmethod
    def create_for(columns: Iterable[base.ColumnReference],
                   projection_type: SelectType = SelectType.Select) -> Select:
        """Full factory method to accompany `star` and `count_star` factory methods.

        This is basically the same as calling the `__init__` method directly.

        Parameters
        ----------
        columns : Iterable[base.ColumnReference]
            The columns that should form the projection
        projection_type : SelectType, optional
            The kind of projection that should be performed, by default `SelectType.Select` which is a plain selection
            without duplicate removal

        Returns
        -------
        Select
            The clause
        """
        target_columns = [BaseProjection.column(column) for column in columns]
        return Select(target_columns, projection_type)

    def __init__(self, targets: BaseProjection | Sequence[BaseProjection],
                 projection_type: SelectType = SelectType.Select) -> None:
        if not targets:
            raise ValueError("At least one target must be specified")
        self._targets = tuple(collection_utils.enlist(targets))
        self._projection_type = projection_type

        hash_val = hash((self._projection_type, self._targets))
        super().__init__(hash_val)

    @property
    def targets(self) -> Sequence[BaseProjection]:
        """Get all projections.

        Returns
        -------
        Sequence[BaseProjection]
            The projections in the order in which they were originally specified
        """
        return self._targets

    @property
    def projection_type(self) -> SelectType:
        """Get the type of projection (with or without duplicate elimination).

        Returns
        -------
        SelectType
            The projection type
        """
        return self._projection_type

    def is_star(self) -> bool:
        """Checks, whether the clause is simply ``SELECT *``.

        Returns
        -------
        bool
            Whether this clause is a ``SELECT *`` clause.
        """
        return len(self._targets) == 1 and self._targets[0] == BaseProjection.star()

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(target.columns() for target in self.targets)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(target.itercolumns() for target in self.targets)

    def tables(self) -> set[base.TableReference]:
        return collection_utils.set_union(target.tables() for target in self.targets)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [target.expression for target in self.targets]

    def output_names(self) -> dict[str, base.ColumnReference]:
        """Output names map the alias of each column to the actual column.

        For example, consider a query ``SELECT R.a AS foo, R.b AS bar FROM R``. Calling `output_names` on this query
        provides the dictionary ``{'foo': R.a, 'bar': R.b}``.

        Currently, this method only works for 1:1 mappings and other aliases are ignored. For example, consider a query
        ``SELECT my_udf(R.a, R.b) AS c FROM R``. Here, a user-defined function is used to combine the values of ``R.a``
        and ``R.b`` to form an output column ``c``. Such a projection is ignored by `output_names`.

        Returns
        -------
        dict[str, base.ColumnReference]
            A mapping from the column target name to the original column.
        """
        output = {}
        for projection in self.targets:
            if not projection.target_name:
                continue
            source_columns = projection.expression.columns()
            if len(source_columns) != 1:
                continue
            output[projection.target_name] = collection_utils.simplify(source_columns)
        return output

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_select_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.projection_type == other.projection_type
                and self.targets == other.targets)

    def __str__(self) -> str:
        select_str = self.projection_type.value
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"


class TableSource(abc.ABC):
    """A table source models a relation that can be scanned by the database system, filtered, joined, ...

    This is what is commonly reffered to as a *table* or a *relation* and forms the basic item of a ``FROM`` clause. In
    an SQL query the items of the ``FROM`` clause can originate from a number of different concepts. In the simplest
    case, this is just a physical table (e.g. ``SELECT * FROM R, S, T WHERE ...``), but other queries might reference
    subqueries or common table expressions in the ``FROM`` clause (e.g.
    ``SELECT * FROM R, (SELECT * FROM S, T WHERE ...) WHERE ...``). This class models the similarities between these
    concepts. Specific sub-classes implement them for the concrete kind of source (e.g. `DirectTableSource` or
    `SubqueryTableSource`).
    """

    @abc.abstractmethod
    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced in the source.

        For plain table sources this will just be the actual table. For more complicated structures, such as subquery
        sources, this will include all tables of the subquery as well.

        Returns
        -------
        set[base.TableReference]
            The tables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all column sthat are referenced in the source.

        For plain table sources this will be empty, but for more complicate structures such as subquery source, this
        will include all columns that are referenced in the subquery.

        Returns
        -------
        set[base.ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all directly contained expressions in the source.

        For plain table sources this will be empty, but for subquery sources, etc. all expressions are returned. Nested
        expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression` interface for
        details).

        Returns
        -------
        Iterable[expr.SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides access to all column in the source.

        For plain table sources this will be empty, but for subquery sources, etc. all expressions are returned. In
        contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n*
        times in this source, it will also be returned *n* times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.

        Returns
        -------
        Iterable[base.ColumnReference]
            All columns exactly in the order in which they are used
        """
        raise NotImplementedError

    def predicates(self) -> preds.QueryPredicates | None:
        """Provides all predicates that are contained in the source.

        For plain table sources this will be ``None``, but for subquery sources, etc. all predicates are returned.

        Returns
        -------
        preds.QueryPredicates | None
            The predicates or ``None`` if the source does not allow predicates or simply does not contain any.
        """
        raise NotImplementedError


class DirectTableSource(TableSource):
    """Models a plain table that is directly referenced in a ``FROM`` clause, e.g. ``R`` in ``SELECT * FROM R, S``.

    Parameters
    ----------
    table : base.TableReference
        The table that is sourced
    """
    def __init__(self, table: base.TableReference) -> None:
        self._table = table

    @property
    def table(self) -> base.TableReference:
        """Get the table that is sourced.

        This can be a virtual table (e.g. for CTEs), but will most commonly be an actual table.

        Returns
        -------
        base.TableReference
            The table.
        """
        return self._table

    def tables(self) -> set[base.TableReference]:
        return {self._table}

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def predicates(self) -> preds.QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return hash(self._table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __repr__(self) -> str:
        return str(self._table)

    def __str__(self) -> str:
        return str(self._table)


class SubqueryTableSource(TableSource):
    """Models subquery that is referenced as a virtual table in the ``FROM`` clause.

    Consider the example query ``SELECT * FROM R, (SELECT * FROM S, T WHERE S.a = T.b) AS s_t WHERE R.c = s_t.a``.
    In this query, the subquery ``s_t`` would be represented as a subquery table source.

    Parameters
    ----------
    query : qal.SqlQuery | expr.SubqueryExpression
        The query that is sourced as a subquery
    target_name : str
        The name under which the subquery should be made available

    Raises
    ------
    ValueError
        If the `target_name` is empty
    """

    def __init__(self, query: qal.SqlQuery | expr.SubqueryExpression, target_name: str) -> None:
        if not target_name:
            raise ValueError("Target name for subquery source is required")
        self._subquery_expression = (query if isinstance(query, expr.SubqueryExpression)
                                     else expr.SubqueryExpression(query))
        self._target_name = target_name
        self._hash_val = hash((self._subquery_expression, self._target_name))

    @property
    def query(self) -> qal.SqlQuery:
        """Get the query that is sourced as a virtual table.

        Returns
        -------
        qal.SqlQuery
            The query
        """
        return self._subquery_expression.query

    @property
    def target_name(self) -> str:
        """Get the name under which the virtual table can be accessed in the actual query.

        Returns
        -------
        str
            The name. This will never be empty.
        """
        return self._target_name

    @property
    def target_table(self) -> base.TableReference:
        """Get the name under which the virtual table can be accessed in the actual query.

        The only difference to `target_name` this return type: this property provides the name as a proper table
        reference, rather than a string.

        Returns
        -------
        base.TableReference
            The table. This will always be a virtual table
        """
        return base.TableReference.create_virtual(self._target_name)

    @property
    def expression(self) -> expr.SubqueryExpression:
        """Get the query that is used to construct the virtual table, as a subquery expression.

        Returns
        -------
        expr.SubqueryExpression
            The subquery.
        """
        return self._subquery_expression

    def tables(self) -> set[base.TableReference]:
        return self._subquery_expression.tables() | {self.target_table}

    def columns(self) -> set[base.ColumnReference]:
        return self._subquery_expression.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [self._subquery_expression]

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self._subquery_expression.itercolumns()

    def predicates(self) -> preds.QueryPredicates | None:
        return self._subquery_expression.query.predicates()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self)) and self._subquery_expression == other._subquery_expression
                and self._target_name == other._target_name)

    def __repr__(self) -> str:
        return str(self._subquery_expression)

    def __str__(self) -> str:
        query_str = self._subquery_expression.query.stringify(trailing_delimiter=False)
        return f"({query_str}) AS {self._target_name}"


class JoinType(enum.Enum):
    """Indicates the type of a join using the explicit ``JOIN`` syntax, e.g. ``OUTER JOIN`` or ``NATURAL JOIN``.

    The names of the individual values should be pretty self-explanatory and correspond entirely to the names in the
    SQL standard.
    """
    InnerJoin = "JOIN"
    OuterJoin = "OUTER JOIN"
    LeftJoin = "LEFT JOIN"
    RightJoin = "RIGHT JOIN"
    CrossJoin = "CROSS JOIN"

    NaturalInnerJoin = "NATURAL JOIN"
    NaturalOuterJoin = "NATURAL OUTER JOIN"
    NaturalLeftJoin = "NATURAL LEFT JOIN"
    NaturalRightJoin = "NATURAL RIGHT JOIN"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.value


class JoinTableSource(TableSource):
    """Models a table that is referenced in a ``FROM`` clause using the explicit ``JOIN`` syntax.

    Such a table source consists of two parts: the actual table source that represents the relation being accessed, as
    well as the condition that is used to join the table source with the other tables.

    Parameters
    ----------
    source : TableSource
        The actual table being sourced
    join_condition : Optional[preds.AbstractPredicate], optional
        The predicate that is used to join the specified table with the other tables of the ``FROM`` clause. For most
        joins this is a required argument in order to create a valid SQL query (e.g. ``LEFT JOIN`` or ``INNER JOIN``),
        but there are some joins without a condition (e.g. ``CROSS JOIN`` and ``NATURAL JOIN``). It is up to the user
        to determine whether a join condition is required for the join in question or not.
    join_type : JoinType, optional
        The specific join that should be performed. Defaults to `JoinType.InnerJoin`.

    Raises
    ------
    ValueError
        If an attempt is made to nest join table sources, i.e. if the `source` is an instance of this class.
    """

    def __init__(self, source: TableSource, join_condition: Optional[preds.AbstractPredicate] = None, *,
                 join_type: JoinType = JoinType.InnerJoin) -> None:
        if isinstance(source, JoinTableSource):
            raise ValueError("JOIN statements cannot have another JOIN statement as source")
        self._source = source
        self._join_condition = join_condition
        self._join_type = join_type if join_condition else JoinType.CrossJoin
        self._hash_val = hash((self._source, self._join_condition, self._join_type))

    @property
    def source(self) -> TableSource:
        """Get the actual table being joined. This can be a proper table or a subquery.

        Returns
        -------
        TableSource
            The table
        """
        return self._source

    @property
    def join_condition(self) -> Optional[preds.AbstractPredicate]:
        """Get the predicate that is used to determine matching tuples from the table.

        This can be ``None`` if the specific `join_type` does not require or allow a join condition (e.g.
        ``NATURAL JOIN``).

        Returns
        -------
        Optional[preds.AbstractPredicate]
            The condition if it is specified, ``None`` otherwise.
        """
        return self._join_condition

    @property
    def join_type(self) -> JoinType:
        """Get the kind of join that should be performed.

        Returns
        -------
        JoinType
            The join type
        """
        return self._join_type

    def tables(self) -> set[base.TableReference]:
        return self._source.tables()

    def columns(self) -> set[base.ColumnReference]:
        condition_columns = self._join_condition.columns() if self._join_condition else set()
        return self._source.columns() | condition_columns

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        source_expressions = list(self._source.iterexpressions())
        condition_expressions = list(self._join_condition.iterexpressions()) if self._join_condition else []
        return source_expressions + condition_expressions

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        source_columns = list(self._source.itercolumns())
        condition_columns = list(self._join_condition.itercolumns()) if self._join_condition else []
        return source_columns + condition_columns

    def predicates(self) -> preds.QueryPredicates | None:
        source_predicates = self._source.predicates()
        condition_predicates = preds.QueryPredicates(self._join_condition) if self._join_condition else None

        if source_predicates and condition_predicates:
            return source_predicates.and_(condition_predicates)
        elif source_predicates:
            return source_predicates
        elif condition_predicates:
            return condition_predicates
        else:
            return None

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self)) and self._source == other._source
                and self._join_condition == other._join_condition
                and self._join_type == other._join_type)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        join_str = str(self.join_type)
        join_prefix = f"{join_str} {self.source}"
        if self.join_condition:
            condition_str = (f"({self.join_condition})" if self.join_condition.is_compound()
                             else str(self.join_condition))
            return join_prefix + f" ON {condition_str}"
        else:
            return join_prefix


class From(BaseClause):
    """The ``FROM`` clause models which tables should be selected and potentially how they are combined.

    A ``FROM`` clause permits arbitrary source items and does not enforce a specific structure or semantic on them.
    This puts the user in charge to generate a valid and meaningful structure. For example, the model allows for the
    first item to be a `JoinTableSource`, even though this is not valid SQL. Likewise, no duplicate checks are
    performed.

    To represent ``FROM`` clauses with a bit more structure, the `ImplicitFromClause` and `ExplicitFromClause`
    subclasses exist and should generally be preffered over direct usage of the raw `From` clause class.

    Parameters
    ----------
    items : TableSource | Iterable[TableSource]
        The tables that should be sourced in the ``FROM`` clause

    Raises
    ------
    ValueError
        If no items are specified
    """
    def __init__(self, items: TableSource | Iterable[TableSource]):
        items = collection_utils.enlist(items)
        if not items:
            raise ValueError("At least one source is required")
        self._items: tuple[TableSource] = tuple(items)
        super().__init__(hash(self._items))

    @property
    def items(self) -> Sequence[TableSource]:
        """Get the tables that are sourced in the ``FROM`` clause

        Returns
        -------
        Sequence[TableSource]
            The sources in exactly the sequence in which they were specified
        """
        return self._items

    def tables(self) -> set[base.TableReference]:
        return collection_utils.set_union(src.tables() for src in self._items)

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(src.columns() for src in self._items)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten(src.iterexpressions() for src in self._items)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(src.itercolumns() for src in self._items)

    def predicates(self) -> preds.QueryPredicates:
        source_predicates = [src.predicates() for src in self._items]
        if not any(source_predicates):
            return preds.QueryPredicates.empty_predicate()
        actual_predicates = [src_pred.root for src_pred in source_predicates if src_pred]
        merged_predicate = preds.CompoundPredicate.create_and(actual_predicates)
        return preds.QueryPredicates(merged_predicate)

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_from_clause(visitor)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._items == other._items

    def __str__(self) -> str:
        fixture = "FROM "
        contents_str = []
        for src in self._items:
            if isinstance(src, JoinTableSource):
                contents_str.append(" " + str(src))
            elif contents_str:
                contents_str.append(", " + str(src))
            else:
                contents_str.append(str(src))
        return fixture + "".join(contents_str)


class ImplicitFromClause(From):
    """Represents a special case of ``FROM`` clause that only allows for pure tables to be selected.

    Specifically, this means that subqueries or explicit joins using the ``JOIN ON`` syntax are not allowed. Just
    plain old ``SELECT ... FROM R, S, T WHERE ...`` queries.

    As a special case, all ``FROM`` clauses that consist of a single (non-subquery) table can be represented as
    implicit clauses.

    Parameters
    ----------
    tables : DirectTableSource | Iterable[DirectTableSource]
        The tables that should be selected
    """

    @staticmethod
    def create_for(tables: base.TableReference | Iterable[base.TableReference]) -> ImplicitFromClause:
        """Shorthand method to create a ``FROM`` clause for a set of table references.

        This saves the user from creating the `DirectTableSource` instances before instantiating a implicit ``FROM``
        clause.

        Parameters
        ----------
        tables : base.TableReference | Iterable[base.TableReference]
            The tables that should be sourced

        Returns
        -------
        ImplicitFromClause
            The ``FROM`` clause
        """
        tables = collection_utils.enlist(tables)
        return ImplicitFromClause([DirectTableSource(tab) for tab in tables])

    def __init__(self, tables: DirectTableSource | Iterable[DirectTableSource]):
        super().__init__(tables)

    def itertables(self) -> Sequence[base.TableReference]:
        """Provides all tables in the ``FROM`` clause exactly in the sequence in which they were specified.

        This utility saves the user from unwrapping all the `DirectTableSource` objects by herself.

        Returns
        -------
        Sequence[base.TableReference]
            The tables.
        """
        return [src.table for src in self.items]


class ExplicitFromClause(From):
    """Represents a special kind of ``FROM`` clause that requires all tables to be joined using the ``JOIN ON`` syntax.

    The tables themselves can be either instances of `DirectTableSource`, or `SubqueryTableSource`.

    Parameters
    ----------
    base_table : DirectTableSource | SubqueryTableSource
        The first table in the ``FROM`` clause. This is the only table that is not part of a ``JOIN ON`` statement.
    joined_tables : Iterable[JoinTableSource]
        All tables that should be joined. At least one such table is required.

    Raises
    ------
    ValueError
        If the `joined_tables` are empty.
    """

    def __init__(self, base_table: DirectTableSource | SubqueryTableSource, joined_tables: Iterable[JoinTableSource]):
        super().__init__([base_table] + list(joined_tables))
        self._base_table = base_table
        self._joined_tables = tuple(joined_tables)
        if not self._joined_tables:
            raise ValueError("At least one joined table expected!")

    @property
    def base_table(self) -> DirectTableSource | SubqueryTableSource:
        """Get the first table that is part of the FROM clause.

        Returns
        -------
        DirectTableSource | SubqueryTableSource
            The table
        """
        return self._base_table

    @property
    def joined_tables(self) -> Sequence[JoinTableSource]:
        """Get all tables that are defined in the ``JOIN ON`` syntax.

        Returns
        -------
        Sequence[JoinTableSource]
            The tables in exactly the same sequence in which they were specified
        """
        return self._joined_tables


class Where(BaseClause):
    """The ``WHERE`` clause specifies conditions that result rows must satisfy.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.

    Parameters
    ----------
    predicate : preds.AbstractPredicate
        The root predicate that specifies all conditions
    """

    def __init__(self, predicate: preds.AbstractPredicate) -> None:
        if not predicate:
            raise ValueError("Predicate must be set")
        self._predicate = predicate
        super().__init__(hash(predicate))

    @property
    def predicate(self) -> preds.AbstractPredicate:
        """Get the root predicate that contains all filters and joins in the ``WHERE`` clause.

        Returns
        -------
        preds.AbstractPredicate
            The condition
        """
        return self._predicate

    def tables(self) -> set[base.TableReference]:
        return self._predicate.tables()

    def columns(self) -> set[base.ColumnReference]:
        return self.predicate.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.predicate.iterexpressions()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.predicate.itercolumns()

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_where_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.predicate == other.predicate

    def __str__(self) -> str:
        return f"WHERE {self.predicate}"


class GroupBy(BaseClause):
    """The ``GROUP BY`` clause combines rows that match a grouping criterion to enable aggregation on these groups.

    Despite their names, all grouped columns can be arbitrary `SqlExpression` instances, rules and restrictions of the SQL
    standard are not enforced by PostBOUND.

    Parameters
    ----------
    group_columns : Sequence[expr.SqlExpression]
        The expressions that should be used to perform the grouping
    distinct : bool, optional
        Whether the grouping should perform duplicate elimination, by default ``False``

    Raises
    ------
    ValueError
        If `group_columns` is empty.
    """

    def __init__(self, group_columns: Sequence[expr.SqlExpression], distinct: bool = False) -> None:
        if not group_columns:
            raise ValueError("At least one group column must be specified")
        self._group_columns = tuple(group_columns)
        self._distinct = distinct

        hash_val = hash((self._group_columns, self._distinct))
        super().__init__(hash_val)

    @property
    def group_columns(self) -> Sequence[expr.SqlExpression]:
        """Get all expressions that should be used to determine the grouping.

        Returns
        -------
        Sequence[expr.SqlExpression]
            The grouping expressions in exactly the sequence in which they were specified.
        """
        return self._group_columns

    @property
    def distinct(self) -> bool:
        """Get whether the grouping should eliminate duplicates.

        Returns
        -------
        bool
            Whether duplicate removal is performed.
        """
        return self._distinct

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(column.columns() for column in self.group_columns)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.group_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(column.itercolumns() for column in self.group_columns)

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_groupby_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.group_columns == other.group_columns and self.distinct == other.distinct)

    def __str__(self) -> str:
        columns_str = ", ".join(str(col) for col in self.group_columns)
        distinct_str = " DISTINCT" if self.distinct else ""
        return f"GROUP BY{distinct_str} {columns_str}"


class Having(BaseClause):
    """The ``HAVING`` clause enables filtering of the groups constructed by a ``GROUP BY`` clause.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.

    The structure of this clause is similar to the `Where` clause, but its scope is different (even though PostBOUND
    does no semantic validation to enforce this): predicates of the ``HAVING`` clause are only checked on entire groups
    of values and have to be valid their, instead of on individual tuples.

    Parameters
    ----------
    condition : preds.AbstractPredicate
        The root predicate that contains all actual conditions
    """

    def __init__(self, condition: preds.AbstractPredicate) -> None:
        if not condition:
            raise ValueError("Condition must be set")
        self._condition = condition
        super().__init__(hash(condition))

    @property
    def condition(self) -> preds.AbstractPredicate:
        """Get the root predicate that is used to form the ``HAVING`` clause.

        Returns
        -------
        preds.AbstractPredicate
            The condition
        """
        return self._condition

    def columns(self) -> set[base.ColumnReference]:
        return self.condition.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.condition.iterexpressions()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.condition.itercolumns()

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_having_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.condition == other.condition

    def __str__(self) -> str:
        return f"HAVING {self.condition}"


class OrderByExpression:
    """The `OrderByExpression` is the fundamental building block for an ``ORDER BY`` clause.

    Each expression consists of the actual column (which might be an arbitrary `SqlExpression`, rules and restrictions
    by the SQL standard are not enforced here) as well as information regarding the ordering of the column. Setting
    this information to `None` falls back to the default interpretation by the target database system.

    Parameters
    ----------
    column : expr.SqlExpression
        The column that should be used for ordering
    ascending : Optional[bool], optional
        Whether the column values should be sorted in ascending order. Defaults to ``None``, which indicates that the
        system-default ordering should be used.
    nulls_first : Optional[bool], optional
        Whether ``NULL`` values should be placed at beginning or at the end of the sorted list. Defaults to ``None``,
        which indicates that the system-default behaviour should be used.
    """

    def __init__(self, column: expr.SqlExpression, ascending: Optional[bool] = None,
                 nulls_first: Optional[bool] = None) -> None:
        if not column:
            raise ValueError("Column must be specified")
        self._column = column
        self._ascending = ascending
        self._nulls_first = nulls_first
        self._hash_val = hash((self._column, self._ascending, self._nulls_first))

    @property
    def column(self) -> expr.SqlExpression:
        """Get the expression used to specify the current grouping.

        In the simplest case this can just be a `ColumnExpression` which sorts directly by the column values. More
        complicated constructs like mathematical expressions over the column values are also possible.

        Returns
        -------
        expr.SqlExpression
            The expression
        """
        return self._column

    @property
    def ascending(self) -> Optional[bool]:
        """Get the desired ordering of the output rows.

        Returns
        -------
        Optional[bool]
            Whether to sort in ascending order. ``None`` indicates that the default behaviour of the system should be
            used.
        """
        return self._ascending

    @property
    def nulls_first(self) -> Optional[bool]:
        """Get where to place ``NULL`` values in the result set.

        Returns
        -------
        Optional[bool]
            Whether to put ``NULL`` values at the beginning of the result set (or at the end). ``None`` indicates that
            the default behaviour of the system should be used.
        """
        return self._nulls_first

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.column == other.column
                and self.ascending == other.ascending
                and self.nulls_first == other.nulls_first)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        ascending_str = "" if self.ascending is None else (" ASC" if self.ascending else " DESC")
        nulls_first = "" if self.nulls_first is None else (" NULLS FIRST " if self.nulls_first else " NULLS LAST")
        return f"{self.column}{ascending_str}{nulls_first}"


class OrderBy(BaseClause):
    """The ``ORDER BY`` clause specifies how result rows should be sorted.

    This clause has a similar structure like a `Select` clause and simply consists of an arbitrary number of
    `OrderByExpression` objects.

    Parameters
    ----------
    expressions : Iterable[OrderByExpression]
        The terms that should be used to determine the ordering. At least one expression is required

    Raises
    ------
    ValueError
        If no `expressions` are provided
    """

    def __init__(self, expressions: Iterable[OrderByExpression]) -> None:
        if not expressions:
            raise ValueError("At least one ORDER BY expression required")
        self._expressions = tuple(expressions)
        super().__init__(hash(self._expressions))

    @property
    def expressions(self) -> Sequence[OrderByExpression]:
        """Get the expressions that form this ``ORDER BY`` clause.

        Returns
        -------
        Sequence[OrderByExpression]
            The individual terms that make up the ordering in exactly the sequence in which they were specified (which
            is the only valid sequence since all other orders could change the ordering of the result set).
        """
        return self._expressions

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(expression.column.columns() for expression in self.expressions)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [expression.column for expression in self.expressions]

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(expression.itercolumns() for expression in self.iterexpressions())

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_orderby_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.expressions == other.expressions

    def __str__(self) -> str:
        return "ORDER BY " + ", ".join(str(order_expr) for order_expr in self.expressions)


class Limit(BaseClause):
    """The ``FETCH FIRST`` or ``LIMIT`` clause restricts the number of output rows returned by the database system.

    Each clause can specify an offset (which is probably only meaningful if there is also an ``ORDER BY`` clause)
    and the actual limit. Notice that although many database systems use a non-standard syntax for this clause, our
    implementation is modelled after the actual SQL standard version (i.e. it produces a ``FETCH ...`` string output).

    Parameters
    ----------
    limit : Optional[int], optional
        The maximum number of tuples to put in the result set. Defaults to ``None`` which indicates that all tuples
        should be returned.
    offset : Optional[int], optional
        The number of tuples that should be skipped from the beginning of the result set. If no `OrderBy` clause is
        defined, this makes the result set's contents non-deterministic (at least in theory). Defaults to ``None``
        which indicates that no tuples should be skipped.

    Raises
    ------
    ValueError
        If neither a `limit`, nor an `offset` are specified
    """

    def __init__(self, *, limit: Optional[int] = None, offset: Optional[int] = None) -> None:
        if limit is None and offset is None:
            raise ValueError("Limit and offset cannot be both unspecified")
        self._limit = limit
        self._offset = offset

        hash_val = hash((self._limit, self._offset))
        super().__init__(hash_val)

    @property
    def limit(self) -> Optional[int]:
        """Get the maximum number of rows in the result set.

        Returns
        -------
        Optional[int]
            The limit or ``None``, if all rows should be returned.
        """
        return self._limit

    @property
    def offset(self) -> Optional[int]:
        """Get the offset within the result set (i.e. number of first rows to skip).

        Returns
        -------
        Optional[int]
            The offset or ``None`` if no rows should be skipped.
        """
        return self._offset

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_limit_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.limit == other.limit and self.offset == other.offset

    def __str__(self) -> str:
        offset_str = f"OFFSET {self.offset} ROWS" if self.offset is not None else ""
        limit_str = f"FETCH FIRST {self.limit} ROWS ONLY" if self.limit is not None else ""
        if offset_str and limit_str:
            return offset_str + " " + limit_str
        elif offset_str:
            return offset_str
        elif limit_str:
            return limit_str
        return ""


VisitorResult = typing.TypeVar("VisitorResult")
"""Return type of the visitor process."""


class ClauseVisitor(abc.ABC, typing.Generic[VisitorResult]):
    """Basic visitor to operate on arbitrary clause lists.

    See Also
    --------
    BaseClause

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_hint_clause(self, clause: Hint) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_explain_clause(self, clause: Explain) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_cte_clause(self, clause: WithQuery) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_select_clause(self, clause: Select) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_from_clause(self, clause: From) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_where_clause(self, clause: Where) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_groupby_clause(self, clause: GroupBy) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_having_clause(self, clause: Having) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_orderby_clause(self, clause: OrderBy) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_limit_clause(self, clause: Limit) -> VisitorResult:
        raise NotImplementedError
