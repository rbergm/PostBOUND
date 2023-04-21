"""Contains the implementation of all supported SQL clauses."""
from __future__ import annotations

import abc
import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable

from postbound.qal import base, expressions as expr, joins, predicates as preds
from postbound.util import collections as collection_utils


# TODO: move away from dataclasses where necessary. Many clauses actually have constraints (non-None / non-empty) objects

class BaseClause(abc.ABC):
    """Basic interface shared by all supported clauses. This is an abstract interface, not a usable clause."""

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced in the clause."""
        return {column.table for column in self.columns() if column.is_bound()}

    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced in the clause."""
        raise NotImplementedError

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all directly contained expressions in this clause.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).
        """
        raise NotImplementedError

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides access to all column in this clause.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced `n`
        times in this clause, it will also be returned `n` times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.
        """
        raise NotImplementedError


@dataclass
class Hint(BaseClause):
    """Hint block of a clause.

    Depending on the SQL dialect, these hints will be placed at different points in the query. Furthermore, the precise
    contents (i.e. syntax and semantic) vary from database system to system.

    Hints are differentiated in two parts:

    - preparatory statements can be executed as valid commands on the database system, e.g. optimizer settings, etc.
    - query hints are the actual hints. Typically, these will be inserted as comments at some place in the query.

    For example, a hint clause for MySQL could look like this:

    ```sql
    SET optimizer_switch = 'block_nested_loop=off';
    SELECT /*+ HASH_JOIN(R S) */ R.a
    FROM R, S, T
    WHERE R.a = S.b AND S.b = T.c
    ```

    This enforces the join between tables `R` and `S` to be executed as a hash join (due to the query hint) and disables
    usage of the block nested-loop join for the entire query (which in this case only affects the join between tables
    `S` and `T`) due to the preparatory `SET optimizer_switch` statement.
    """
    preparatory_statements: str = ""
    query_hints: str = ""

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def __hash__(self) -> int:
        return hash((self.preparatory_statements, self.query_hints))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.preparatory_statements == other.preparatory_statements
                and self.query_hints == other.query_hints)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.preparatory_statements and self.query_hints:
            return self.preparatory_statements + "\n" + self.query_hints
        elif self.preparatory_statements:
            return self.preparatory_statements
        return self.query_hints


@dataclass
class Explain(BaseClause):
    """EXPLAIN block of a query.

    EXPLAIN queries change the execution mode of a query. Instead of focusing on the actual query result, an EXPLAIN
    query produces information about the internal processes of the database system. Typically, this includes which
    execution plan the DBS would choose for the query. Additionally, EXPLAIN ANALYZE (as for example supported by
    Postgres) provides the query plan and executes the actual query. The returned plan is then annotated by how the
    optimizer predictions match reality. Furthermore, such ANALYZE plans typically also contain some runtime statistics
    such as runtime of certain operators.

    The precise syntax and semantic of an EXPLAIN statement depends on the actual DBS. The Explain clause object
    is modeled after Postgres.
    """
    analyze: bool = False
    format: str | None = None

    @staticmethod
    def explain_analyze(format_type: str = "JSON") -> Explain:
        """Constructs an EXPLAIN ANALYZE clause with the specified output format."""
        return Explain(True, format_type)

    @staticmethod
    def plan(format_type: str = "JSON") -> Explain:
        """Constructs a pure EXPLAIN clause (i.e. without ANALYZE) with the specified output format."""
        return Explain(False, format_type)

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def __hash__(self) -> int:
        return hash((self.analyze, self.format))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.analyze == other.analyze and self.format == other.format

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        explain_prefix = "EXPLAIN"
        explain_body = ""
        if self.analyze and self.format:
            explain_body = f" (ANALYZE, FORMAT {self.format})"
        elif self.analyze:
            explain_body = " ANALYZE"
        elif self.format:
            explain_body = f" (FORMAT {self.format})"
        return explain_prefix + explain_body


@dataclass
class BaseProjection:
    """The `BaseProjection` forms the fundamental ingredient for a SELECT clause.

    Each SELECT clause is composed of at least one `BaseProjection`. Each projection can be an arbitrary
    `SqlExpression` (rules and restrictions of the SQL standard are not enforced here). In addition, each projection
    can receive a target name as in `SELECT foo AS f FROM bar`.
    """
    expression: expr.SqlExpression
    target_name: str = ""

    @staticmethod
    def count_star() -> BaseProjection:
        """Shortcut to create a COUNT(*) projection."""
        return BaseProjection(expr.FunctionExpression("count", [expr.StarExpression()]))

    @staticmethod
    def star() -> BaseProjection:
        """Shortcut to create a * projection."""
        return BaseProjection(expr.StarExpression())

    @staticmethod
    def column(col: base.ColumnReference, target_name: str = "") -> BaseProjection:
        """Shortcut to create a projection for the given column."""
        return BaseProjection(expr.ColumnExpression(col), target_name)

    def columns(self) -> set[base.ColumnReference]:
        return self.expression.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.expression.itercolumns()

    def tables(self) -> set[base.TableReference]:
        return self.expression.tables()

    def __hash__(self) -> int:
        return hash((self.expression, self.target_name))

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
    """Indicates the specific type of the SELECT clause."""
    Select = "SELECT"
    SelectDistinct = "SELECT DISTINCT"


class Select(BaseClause):
    """The SELECT clause of a query.

    This is the only required part of a query. Everything else is optional and can be left out. (Notice that PostBOUND
    is focused on SPJ-queries, hence there are no INSERT, UPDATE, or DELETE queries)

    A SELECT clause simply consists of a number of individual projections (see `BaseProjection`), the `targets`.
    """

    @staticmethod
    def count_star() -> Select:
        """Shortcut to create a SELECT COUNT(*) clause."""
        return Select(BaseProjection.count_star())

    @staticmethod
    def star() -> Select:
        """Shortcut to create a SELECT * clause."""
        return Select(BaseProjection.star())

    def __init__(self, targets: BaseProjection | list[BaseProjection],
                 projection_type: SelectType = SelectType.Select) -> None:
        self.targets = tuple(collection_utils.enlist(targets))
        self.projection_type = projection_type

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

        For example, consider a query `SELECT R.a AS foo, R.b AS bar FROM R`. Calling `output_names` on this query
        provides the dictionary `{'foo': R.a, 'bar': R.b}`.

        Currently, this method only works for 1:1 mappings and other aliases are ignored. For example, consider a query
        `SELECT my_udf(R.a, R.b) AS c FROM R`. Here, a user-defined function is used to combine the values of `R.a` and
        `R.b` to form an output column `c`. Such a projection is ignored by `output_names`.
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

    def __hash__(self) -> int:
        return hash((self.projection_type, self.targets))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.projection_type == other.projection_type
                and self.targets == other.targets)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        select_str = self.projection_type.value
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"


class From(BaseClause, abc.ABC):
    """FROM clause of the query.

    PostBOUND distinguishes between two types of FROM clauses:
    - implicit FROM clauses simply list all referenced tables as in `SELECT * FROM R, S, T WHERE ...`
    - explicit FROM clauses on the other hand combine the source using the `JOIN ON` syntax as in
    `SELECT * FROM R JOIN S ON ... JOIN T ON ... WHERE ...`

    Note that these types of clauses are mutually exclusive for PostBOUND even though a combination of implicit
    references and JOIN statements could be used in a valid SQL query. I.e., a query can only be either implicit or
    explicit. This behaviour might change in the future to allow for a mixture of both types of references.
    """

    @abc.abstractmethod
    def predicates(self) -> preds.QueryPredicates | None:
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class ImplicitFromClause(From):
    """The implicit FROM clause lists all referenced tables without specifying their relation.

    See `FromClause` for details.

    One limitation of implicit FROM clauses is that currently they may only be composed of tables and not subqueries.
    """

    # TODO: we could also have subqueries in an implicit from clause!

    def __init__(self, tables: base.TableReference | list[base.TableReference] | None = None):
        self._tables = tuple(collection_utils.enlist(tables)) if tables is not None else ()

    def tables(self) -> set[base.TableReference]:
        return set(self._tables)

    def itertables(self) -> Iterable[base.TableReference]:
        return self._tables

    def predicates(self) -> preds.QueryPredicates | None:
        return None

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def __hash__(self) -> int:
        return hash(self._tables)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._tables == other._tables

    def __str__(self) -> str:
        if not self._tables:
            return "[NO TABLES]"
        return "FROM " + ", ".join(str(table) for table in self._tables)


class ExplicitFromClause(From):
    """The explicit FROM clause lists all referenced tables and subqueries using the JOIN ON syntax.

    See `FromClause` for details.
    """

    def __init__(self, base_table: base.TableReference, joined_tables: list[joins.Join]):
        self.base_table = base_table
        self.joined_tables = tuple(joined_tables)

    def tables(self) -> set[base.TableReference]:
        all_tables = [self.base_table]
        for join in self.joined_tables:
            all_tables.extend(join.tables())
        return set(all_tables)

    def predicates(self) -> preds.QueryPredicates | None:
        predicate_handler = preds.DefaultPredicateHandler
        all_predicates = predicate_handler.empty_predicate()
        for join in self.joined_tables:
            if isinstance(join, joins.TableJoin):
                if join.join_condition:
                    all_predicates = all_predicates.and_(join.join_condition)
                continue

            if not isinstance(join, joins.SubqueryJoin):
                TypeError("Unknown join type: " + str(type(join)))
            subquery_join: joins.SubqueryJoin = join

            subquery_predicates = subquery_join.subquery.predicates()
            if subquery_predicates:
                all_predicates = all_predicates.and_(subquery_predicates)
            if subquery_join.join_condition:
                all_predicates = all_predicates.and_(subquery_join.join_condition)

        return all_predicates

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(join.columns() for join in self.joined_tables)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten(join.iterexpressions() for join in self.joined_tables)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(join.itercolumns() for join in self.joined_tables)

    def __hash__(self) -> int:
        return hash((self.base_table, self.joined_tables))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.base_table == other.base_table
                and self.joined_tables == other.joined_tables)

    def __str__(self) -> str:
        return f"FROM {self.base_table} " + " ".join(str(join) for join in self.joined_tables)


@dataclass
class Where(BaseClause):
    """The WHERE clause specifies conditions that result rows must satisfy.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.
    """
    predicate: preds.AbstractPredicate

    def columns(self) -> set[base.ColumnReference]:
        return self.predicate.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.predicate.iterexpressions()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.predicate.itercolumns()

    def __hash__(self) -> int:
        return hash(self.predicate)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.predicate == other.predicate

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"WHERE {self.predicate}"


@dataclass
class GroupBy(BaseClause):
    """The GROUP BY clause combines rows that match a grouping criterion to enable aggregation on these groups.

    All grouped columns can be arbitrary `SqlExpression`s, rules and restrictions of the SQL standard are not enforced
    by PostBOUND.
    """

    def __init__(self, group_columns: Sequence[expr.SqlExpression], distinct: bool = False) -> None:
        self.group_columns = tuple(group_columns)
        self.distinct = distinct

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(column.columns() for column in self.group_columns)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.group_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(column.itercolumns() for column in self.group_columns)

    def __hash__(self) -> int:
        return hash((self.group_columns, self.distinct))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.group_columns == other.group_columns and self.distinct == other.distinct)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        columns_str = ", ".join(str(col) for col in self.group_columns)
        distinct_str = " DISTINCT" if self.distinct else ""
        return f"GROUP BY{distinct_str} {columns_str}"


@dataclass
class Having(BaseClause):
    """The HAVING clause specifies conditions that have to be met on the groups constructed by a GROUP BY clause.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.
    """
    condition: preds.AbstractPredicate

    def columns(self) -> set[base.ColumnReference]:
        return self.condition.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.condition.iterexpressions()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.condition.itercolumns()

    def __hash__(self) -> int:
        return hash(self.condition)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.condition == other.condition

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"HAVING {self.condition}"


@dataclass
class OrderByExpression:
    """The `OrderByExpression` is the fundamental ingredient for an ORDER BY clause.

    Each expression consists of the actual column (which might be an arbitrary `SqlExpression`, rules and restrictions
    by the SQL standard are not enforced here) as well as information regarding the ordering of the column. Setting
    such information to `None` falls back to the default interpretation by the target database system.
    """

    column: expr.SqlExpression
    ascending: bool | None = None
    nulls_first: bool | None = None

    def __hash__(self) -> int:
        return hash((self.column, self.ascending, self.nulls_first))

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
    """The ORDER BY clause specifies how result rows should be sorted.

    It consists of an arbitrary number of `OrderByExpression`s.
    """

    def __init__(self, expressions: list[OrderByExpression]) -> None:
        if not expressions:
            raise ValueError("At least one ORDER BY expression required")
        self.expressions = tuple(expressions)

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(expression.column.columns() for expression in self.expressions)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [expression.column for expression in self.expressions]

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(expression.itercolumns() for expression in self.iterexpressions())

    def __hash__(self) -> int:
        return hash(self.expressions)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.expressions == other.expressions

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "ORDER BY " + ", ".join(str(order_expr) for order_expr in self.expressions)


class Limit(BaseClause):
    """The LIMIT clause restricts the number of output rows returned by the database system.

    Each clause can specify an OFFSET (which is probably only meaningful if there is also an ORDER BY clause) and the
    actual LIMIT. Note that some database systems might use a non-standard syntax for such clauses.
    """

    def __init__(self, *, limit: int | None = None, offset: int | None = None) -> None:
        if limit is None and offset is None:
            raise ValueError("LIMIT and OFFSET cannot be both unspecified")
        self.limit = limit
        self.offset = offset

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def __hash__(self) -> int:
        return hash((self.limit, self.offset))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.limit == other.limit and self.offset == other.offset

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        limit_str = f"LIMIT {self.limit}" if self.limit is not None else ""
        offset_str = f"OFFSET {self.offset}" if self.offset is not None else ""
        return limit_str + offset_str
