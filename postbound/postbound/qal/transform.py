"""`transform` provides utilities to generate SQL queries from other queries."""

from __future__ import annotations

import typing
from typing import Iterable

from postbound.db import db
from postbound.qal import qal, base

_Q = typing.TypeVar("_Q", bound=qal.SqlQuery)


def implicit_to_explicit(source_query: qal.ImplicitSqlQuery) -> qal.ExplicitSqlQuery:
    pass


def explicit_to_implicit(source_query: qal.ExplicitSqlQuery) -> qal.ImplicitSqlQuery:
    pass


def extract_query_fragment(source_query: qal.SqlQuery,
                           referenced_tables: Iterable[base.TableReference]) -> qal.SqlQuery:
    pass


def as_count_star_query(source_query: qal.SqlQuery) -> qal.SqlQuery:
    pass


def rename_table(source_query: qal.SqlQuery, from_table: base.TableReference, target_table: base.TableReference, *,
                 prefix_column_names: bool = False) -> qal.SqlQuery:
    pass


def bind_columns(query: qal.SqlQuery, *, with_schema: bool = True, db_schema: db.DatabaseSchema | None = None) -> None:
    """Queries the table metadata to obtain additional information about the referenced columns.

    The retrieved information includes type information for all columns and the tables that contain the columns.
    """
    if not query.predicates():
        return

    alias_map = {table.alias: table for table in query.tables() if table.alias}
    unbound_tables = [table for table in query.tables() if not table.alias]
    unbound_columns = []
    for column in query.predicates().root().columns():
        if not column.table:
            unbound_columns.append(column)
        elif not column.table.full_name and column.table.alias in alias_map:
            column.table.full_name = alias_map[column.table.alias].full_name

    if with_schema:
        db_schema = db_schema if db_schema else db.DatabasePool.get_instance().current_database()
        for column in unbound_columns:
            column.table = db_schema.lookup_column(column, unbound_tables)
