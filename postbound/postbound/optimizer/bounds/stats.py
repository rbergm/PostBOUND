from __future__ import annotations

from postbound.qal import base, qal
from postbound.optimizer import data


class UpperBoundsContainer:

    def __init__(self) -> None:
        self.base_table_estimates: dict[base.TableReference, int] = {}
        self.upper_bounds: dict[base.TableReference | data.JoinTree, int] = {}

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        pass
