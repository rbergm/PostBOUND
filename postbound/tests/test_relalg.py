"""Tests for the relalg module, specifically for parsing queries into relational algebra.

Notice that these tests merely check that the queries are parsed without errors. They do not check the relational algebra
DAGs for correctness. This is because right now we do not have a good and manageable way to check the correctness of these
structures.
"""
from __future__ import annotations

import unittest

from postbound.qal import parser, relalg


class RelalgParserTests(unittest.TestCase):
    def test_q1(self):
        # Query Q1 from Neumann, Kemper: "Unnesting Arbitrary Queries", BTW'15
        q1 = parser.parse_query("""
                                select s.name, e.course
                                from students s, exams e
                                where s.id = e.sid and
                                    e.grade = (select min(e2.grade)
                                                from exams e2
                                                where s.id=e2.sid)
                                """)
        relalg.parse_relalg(q1)

    def test_q2(self):
        # Query Q2 from Neumann, Kemper: "Unnesting Arbitrary Queries", BTW'15
        q2 = parser.parse_query("""
                                select s.name, e.course
                                from students s, exams e
                                where s.id=e.sid and
                                    (s.major = 'CS' or s.major = 'Games Eng') and
                                    e.grade >= (select avg(e2.grade) + 1
                                                from exams e2
                                                where s.id = e2.sid or
                                                    (e2.curriculum = s.major and
                                                    s.year > e2.date))
                                """)
        relalg.parse_relalg(q2)
