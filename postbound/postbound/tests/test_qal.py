"""Tests for PostBOUND's query abstraction layer.

The tests here mainly act as regression tests to ensure that SqlQuery objects provide the correct information if
queried for joins, etc.
"""

import sys
import unittest

sys.path.append("../../")

from postbound.qal import base, parser

from postbound.tests import regression_suite


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
        self.assertTrue(len(parsed_query.predicates().joins()) == 1, "Should detect 1 join")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = parser.parse_query(subquery_join_query)
        self.assertTrue(len(parsed_query.predicates().joins()) == 0, "Should treat dependent subqueries as filters")

    def test_filter_detection(self) -> None:
        simple_query = "SELECT * FROM R, S WHERE R.a = 42 AND R.b = S.c"
        parsed_query = parser.parse_query(simple_query)
        self.assertTrue(len(parsed_query.predicates().filters()) == 1, "Should detect 1 filter")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = parser.parse_query(subquery_join_query)
        self.assertTrue(len(parsed_query.predicates().filters()) == 1,
                        "Should detect filters for dependent subquery")

        independent_subquery_query = "SELECT * FROM R WHERE R.a = (SELECT MIN(S.b) FROM S)"
        parsed_query = parser.parse_query(independent_subquery_query)
        self.assertTrue(len(parsed_query.predicates().filters()) == 1,
                        "Should detect 1 filter for independent subquery")


class PredicateTests(unittest.TestCase):
    def test_binary_predicate_joins(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a = S.b"
        with self.subTest("Direct equi join", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a < S.b"
        with self.subTest("Direct non-equi join", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = 42"
        with self.subTest("Direct filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE some_udf(R.a, S.b) = 42"
        with self.subTest("Join in UDF", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = some_udf(S.b)"
        with self.subTest("Join with UDF result", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE some_udf(R.a) = 42"
        with self.subTest("Filter with UDF", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a = S.b::integer"
        with self.subTest("Join with casted value", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = (SELECT MIN(T.c) FROM T)"
        with self.subTest("Filter with subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a = (SELECT MIN(T.c) FROM T WHERE T.c = S.b)"
        with self.subTest("Filter with dependent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_between_predicate(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND 42"
        with self.subTest("Direct BETWEEN filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND S.b"
        with self.subTest("Direct BETWEEN join, end", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN S.b AND S42"
        with self.subTest("Direct BETWEEN join, start", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN S.b AND S.b + 42"
        with self.subTest("Direct BETWEEN join, both ends", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND (SELECT MIN(T.c) FROM T)"
        with self.subTest("BETWEEN filter with subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_in_predicate(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a IN (1, 2, 3)"
        with self.subTest("Direct IN filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a IN (1, S.b, 3)"
        with self.subTest("Join in values list", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a IN (SELECT T.b FROM T WHERE T.c = 42)"
        with self.subTest("IN filter for independent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

            query = "SELECT * FROM R, S WHERE R.a IN (SELECT T.b FROM T WHERE T.c = T.d)"
            with self.subTest("IN filter for dependent subquery", query=query):
                parsed = parser.parse_query(query)
                self.assertTrue(len(parsed.predicates().joins()) == 0)
                self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_unary_predicate(self) -> None:
        query = "SELECT * FROM R WHERE EXISTS (SELECT * FROM S WHERE R.a = S.b)"
        with self.subTest("EXISTS for dependent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R WHERE NOT EXISTS (SELECT * FROM S WHERE R.a = S.b)"
        with self.subTest("NOT EXISTS for dependent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE my_udf(R.a)"
        with self.subTest("Unary UDF filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE my_udf(R.a, S.b)"
        with self.subTest("Unary UDF join", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)


class TransformationTests(unittest.TestCase):
    def test_column_binding(self) -> None:
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        col_r_b = base.ColumnReference("b", tab_r)

        tab_s = base.TableReference("S")
        col_s_c = base.ColumnReference("c", tab_s)

        query = "SELECT R.a FROM R WHERE R.b = 42"
        with self.subTest("Simple binding through full name", query=query):
            parsed = parser.parse_query(query, bind_columns=False)  # bind_columns refers to live binding
            self.assertSetEqual(parsed.select_clause.columns(), {col_r_a})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_b})

        query = "SELECT S.c FROM R, S WHERE R.a = S.c AND R.b = 42"
        with self.subTest("Simple binding in join", query=query):
            parsed = parser.parse_query(query, bind_columns=False)  # bind_columns refers to live binding
            self.assertSetEqual(parsed.select_clause.columns(), {col_s_c})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_a, col_r_b, col_s_c})


class ParserTests(regression_suite.QueryTestCase):
    def test_parse_subquery_without_predicates(self) -> None:
        query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S)"
        parsed = parser.parse_query(query)
        self.assertTrue(len(parsed.subqueries()) == 1, "Should detect 1 subquery")

    def test_count_distinct(self) -> None:
        query = "SELECT COUNT(DISTINCT *) FROM R"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format COUNT(DISTINCT *) correctly.")

    def test_is_predicate(self) -> None:
        query = "SELECT * FROM R WHERE R.a IS NULL"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format IS NULL correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter for IS NULL")

        query = "SELECT * FROM R WHERE R.a IS NOT NULL"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format IS NOT NULL correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter for IS NOT NULL")

    def test_unary_udf_filter(self) -> None:
        query = "SELECT * FROM R WHERE my_udf(R.a)"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format unary UDF filter correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter for unary UDF filter")


class RegressionTests(unittest.TestCase):
    # No regressions so far!
    pass