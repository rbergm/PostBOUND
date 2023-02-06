""""""
from __future__ import annotations

import abc
import enum


class ColumnType(abc.ABC):
    pass


class IntegerColumn(ColumnType):
    pass


class TextColumn(ColumnType):
    pass


class VarCharColumn(ColumnType):
    def __init__(self, length: int) -> None:
        self.length = length


class TimestampColumn(ColumnType):
    def __init__(self, with_timezone: bool = False) -> None:
        self.with_timezone = with_timezone


class ColumnTypes(enum.Enum):
    INTEGER = IntegerColumn
    TEXT = TextColumn
    VARCHAR = VarCharColumn
    TIMESTAMP = TimestampColumn
