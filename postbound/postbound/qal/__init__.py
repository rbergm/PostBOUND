"""
Contains the basic **query abstraction layer** to conveniently operate on SQL queries on a logical level.

The most important features of the qal are:
1. parsing query strings into qal objects
2. providing access to underlying query features such as referenced tables, aliases or predicates
3. formatting qal objects back to strings
"""
