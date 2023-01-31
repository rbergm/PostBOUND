"""`transform` provides utilities to generate SQL queries from other queries."""

from typing import Iterable

from postbound.db import db
from postbound.qal import qal, base


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


# TODO: binding info should potentially be re-usable? Return tuple of bound query + binding info instead?
def bind_columns(source_query: qal.SqlQuery, dbs: db.DatabaseSchema) -> qal.SqlQuery:
    """Queries the table metadata to obtain additional information about the referenced columns.

    The retrieved information includes type information for all columns and the tables that contain the columns.
    """
    pass
