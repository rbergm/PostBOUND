"""Shortcuts provide simple methods to generate instances of different PostBOUND objects, mostly for REPL contexts."""
from __future__ import annotations

from postbound.qal import base, qal, parser


def tab(table: str) -> base.TableReference:
    """Creates a table instance.

    Parameters
    ----------
    table : str
        The name and/or alias of the table. Supported formats include ``"table_name"`` and ``"table_name alias"``

    Returns
    -------
    base.TableReference
        The resulting table. This will never be a virtual table.
    """
    if " " in table:
        full_name, alias = table.split(" ")
        return base.TableReference(full_name, alias)
    else:
        return base.TableReference(table)


def col(column: str) -> base.ColumnReference:
    """Creates a column instance.

    Parameters
    ----------
    column : str
        The name and/or table of the column. Supported formats include ``"column_name"`` and ``"table_name.column_name"``

    Returns
    -------
    base.ColumnReference
        The resulting column. If a table name is included before the ``.``, it will be parsed according to the rules of `tab()`.
    """
    if "." in column:
        table_name, column_name = column.split(".")
        return base.ColumnReference(column_name, tab(table_name))
    else:
        return base.ColumnReference(column)


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
        `qal.parser.parse_query`.

    See Also
    --------
    qal.parser.parse_query
    """
    return parser.parse_query(query)
