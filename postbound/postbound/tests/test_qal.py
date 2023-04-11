"""Tests for PostBOUND's query abstraction layer.

The tests here mainly act as regression tests to ensure that SqlQuery objects provide the correct information if
queried for joins, etc.
"""

import sys
import unittest

sys.path.append("../../")

from postbound.qal import parser


class SqlQueryTests(unittest.TestCase):
    def test_dependent_subqueries(self) -> None:
        dependent_query = "SELECT * FROM R WHERE R.a = S.b"
        parsed_query = parser.parse_query(dependent_query)
        self.assertTrue(parsed_query.is_dependent(), "Dependent query not recognized as dependent")

        independent_query = "SELECT * FROM R WHERE R.a = 42"
        parsed_query = parser.parse_query(independent_query)
        self.assertFalse(parsed_query.is_dependent(), "Independent query recognized as dependent")

    def test_join_detection(self) -> None:
        simple_query = "SELECT * FROM R, S WHERE R.a = 42 AND R.b = S.c"
        parsed_query = parser.parse_query(simple_query)
        self.assertTrue(len(list(parsed_query.predicates().joins())) == 1, "Should detect 1 join")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = parser.parse_query(subquery_join_query)
        self.assertTrue(len(list(parsed_query.predicates().joins())) == 1,
                        "Should detect 1 join for dependent subquery")

    def test_filter_detection(self) -> None:
        simple_query = "SELECT * FROM R, S WHERE R.a = 42 AND R.b = S.c"
        parsed_query = parser.parse_query(simple_query)
        self.assertTrue(len(list(parsed_query.predicates().filters())) == 1, "Should detect 1 filter")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = parser.parse_query(subquery_join_query)
        self.assertTrue(len(list(parsed_query.predicates().filters())) == 1,
                        "Should not detect filters for dependent subquery")

        independent_subquery_query = "SELECT * FROM R WHERE R.a = (SELECT MIN(S.b) FROM S)"
        parsed_query = parser.parse_query(independent_subquery_query)
        self.assertTrue(len(list(parsed_query.predicates().filters())) == 1,
                        "Should detect 1 filter for independent subquery")


class TransformationTests(unittest.TestCase):
    def test_column_binding(self) -> None:
        pass


class ParserTests(unittest.TestCase):
    def test_parse_subquery_without_predicates(self) -> None:
        query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S)"
        parsed = parser.parse_query(query)
        self.assertTrue(len(list(parsed.predicates().joins())) == 1)


class RegressionTests(unittest.TestCase):
    # No regressions so far!
    pass
