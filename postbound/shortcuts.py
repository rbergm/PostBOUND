"""Shortcuts provide simple methods to generate instances of different PostBOUND objects, mostly for REPL contexts."""

from __future__ import annotations

from . import qal
from ._core import ColumnReference, TableReference


def tab(table: str) -> TableReference:
    """Creates a table instance.

    Parameters
    ----------
    table : str
        The name and/or alias of the table. Supported formats include ``"table_name"`` and ``"table_name alias"``

    Returns
    -------
    TableReference
        The resulting table. This will never be a virtual table.
    """
    if " " in table:
        full_name, alias = table.split(" ")
        return TableReference(full_name, alias)
    else:
        return TableReference(table)


def col(column: str) -> ColumnReference:
    """Creates a column instance.

    Parameters
    ----------
    column : str
        The name and/or table of the column. Supported formats include ``"column_name"`` and ``"table_name.column_name"``

    Returns
    -------
    ColumnReference
        The resulting column. If a table name is included before the ``.``, it will be parsed according to the rules of
        `tab()`.
    """
    if "." in column:
        table_name, column_name = column.split(".")
        return ColumnReference(column_name, tab(table_name))
    else:
        return ColumnReference(column)


def q(query: str) -> qal.SqlQuery:
    """Parses the given SQL query.

    This is really just a shortcut to calling importing and calling the parser module.

    Parameters
    ----------
    query : str
        The SQL query to parse

    Returns
    -------
    qal.SqlQuery
        A QAL query object corresponding to the given input query. Errors can be produced according to the documentation of
        `qal.parse_query`.

    See Also
    --------
    qal.parse_query
    """
    return qal.parse_query(query)
