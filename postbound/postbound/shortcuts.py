"""Shortcuts provide simple methods to generate instances of different PostBOUND objects, mostly for REPL contexts."""
from __future__ import annotations

from postbound.qal import base, qal, parser


def tab(table: str) -> base.TableReference:
    """Creates a table instance."""
    if " " in table:
        full_name, alias = table.split(" ")
        return base.TableReference(full_name, alias)
    else:
        return base.TableReference(table)


def col(column: str) -> base.ColumnReference:
    """Creates a column instance."""
    if "." in column:
        table_name, column_name = column.split(".")
        return base.ColumnReference(column_name, tab(table_name))
    else:
        return base.ColumnReference(column)


def q(query: str) -> qal.SqlQuery:
    """Parses the given SQL query."""
    return parser.parse_query(query)
