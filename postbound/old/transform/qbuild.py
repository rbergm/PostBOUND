
import abc
from transform import db, mosp


class AbstractClauseBuilder(abc.ABC):
    def __init__(self, query_builder: "QueryBuilder"):
        self._query_builder = query_builder

    def end(self) -> "QueryBuilder":
        return self._query_builder


class SelectClauseBuilder(AbstractClauseBuilder):
    def __init__(self, query_builder: "QueryBuilder"):
        super().__init__(query_builder)

    def star(self) -> "SelectClauseBuilder":
        pass

    def attribute(self, attribute: db.AttributeRef) -> "SelectClauseBuilder":
        pass

    def count(self, attribute: db.AttributeRef) -> "SelectClauseBuilder":
        pass

    def count_star(self) -> "SelectClauseBuilder":
        pass


class FromClauseBuilder(AbstractClauseBuilder):
    def __init__(self, query_builder: "QueryBuilder"):
        super().__init__(query_builder)

    def table(self, table: db.TableRef) -> "FromClauseBuilder":
        pass


class WhereClauseBuilder(AbstractClauseBuilder):
    def __init__(self, query_builder: "QueryBuilder"):
        super().__init__(query_builder)


class QueryBuilder:
    def __init__(self, *, dbs: db.DBSchema = db.DBSchema.get_instance()):
        pass

    def select_clause(self) -> SelectClauseBuilder:
        pass

    def from_clause(self) -> FromClauseBuilder:
        pass

    def where_clause(self) -> WhereClauseBuilder:
        pass

    def mosp(self) -> mosp.MospQuery:
        pass

    def text(self) -> str:
        return self.mosp().text()

    def __repr__(self) -> str:
        return f"QueryBuilder ({self.text()})"

    def __str__(self) -> str:
        return self.text()
