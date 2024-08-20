PostBOUND's implementation of the Cardinality Estimation Benchmark
==================================================================

The Cardinality Estimation Benchmark (CEB) [1]_ provides a powerful framework to generate large workloads of SQL queries based
on simple templates. The query generator splits the templates into sections for individual (groups of) predicates and allows
these predicates to sample correlated values together, introduce dependencies to other predicate values, etc. However, the
templates and their semantics are sadly not well-specified and its often unclear how specific properties interact and influence
each other. Therefore, we ship our own generator implementation inspired by the CEB as part of the PostBOUND framework. In
addition to the actual implementation, we also provide a detailed specification of the templates and their properties in the
following section.

Generator Templates
-------------------

Each template is specified in a single TOML-file. A template has to contain the following sections:

- A leading ``title`` key of type String. It specifies the name of the current template and is mainly used for
  logging/debugging purposes.
- A ``base_sql`` table. Within the table, the ``sql`` key contains the final SQL query to generate (including the placeholder
  values). The ``table_aliases`` is a required inline-table that maps each alias to a fully-qualified name. If no aliases are
  given, the values can simply be empty strings. For example, for ``SELECT * FROM title t``, ``table_aliases`` should be
  ``{ t = "title" }``, whereas for ``SELECT * FROM title``, it should be ``{title = ""}``. This slight inconsistency is
  necessary to accomodate for tables that are referenced with multiple aliases as well as tables that are referenced without an
  alias within the same query (which still is valid SQL..).

The base SQL query can contain placeholders in the form of ``<<placeholder>>``. These placeholders are replaced by the actual
predicate values (see below).

Within the ``base_sql`` table, an array of tables called ``predicates`` contains the actual instructions to generate appropriate
values for the placeholders. A predicate is structured as follows:

+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Key                 | Key type                   | Allowed values                                            | Dependencies                                              | Description                                                                                                                                                                                                                             |
+=====================+============================+===========================================================+===========================================================+=========================================================================================================================================================================================================================================+
| ``name``            | String, required           | *arbitrary*                                               | *none*                                                    | An alias for the current predicate. The name can be referenced as a dependency for other predicates.                                                                                                                                    |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``dependencies``    | Array of strings, optional | ``name`` properties of other predicates                   | *none*                                                    | Specifies that values computed in the referenced predicates have to injected into the query that computes the current predicate values. Dependencies are assumed to form the same predicates as in the base SQL.                        |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``keys``            | Array of strings, required | Placeholders from the `base_sql` query.                   | *none*                                                    | The placeholders of the base SQL whose values are computed by this predicate. Leading ``<<`` and trailing ``>>`` are optional but *should* be used consistently.                                                                        |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``columns``         | Array of strings, required | (Qualified) column names                                  | *none*                                                    | Output columns in the SQL query of this predicate that compute the given ``keys``. The i-th column value corresponds to the i-th key-value. This is required to determine the column types and escape selected values appropriately.    |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``pred_type``       | String, required           | ``=``, ``<``, ``>``, ``<=``, ``>=``, ``LIKE``, ``IN``     | *none*                                                    | The predicate to be inserted into the base SQL. This key ensures that the selected values are escaped properly.                                                                                                                         |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``sampling_method`` | String, required           | ``uniform`` or ``weighted``                               | *none*                                                    | ``uniform`` means that all column values are selected with equal probability, ignoring duplicate values. ``weighted`` means that values that appear multiple times have a higher chance of being selected.                              |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``type``            | String, required           | ``list`` or ``sql``                                       | *none*                                                    | ``list`` provides an explicit array of allowed values, whereas ``sql`` computes the value dynamically using a custom query.                                                                                                             |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``options``         | Array, optional            | *arbitrary*                                               | Required for ``type = list`` templates, otherwise ignored | The values to pick from. As a special case, Nested arrays can be used if multiple placeholders should be filled at the same time. For example, ``date >= <<D1>> AND date <= <<D2>>`` could be generated from a list ``[[1,2], [3,4]]``. |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``sql``             | String, optional           | SQL query.                                                | Required for ``type = sql``, otherwise ignored            | The SQL query that computes the ``columns``. It can contain placeholders that are computed by the ``dependencies``. These placeholders are substitude before execution.                                                                 |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``min_samples``     | Integer, optional          | *arbitrary*                                               | Ignored unless ``pred_type = IN``                         | The minimum number of values to insert into the IN predicate, defaults to 1.                                                                                                                                                            |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``max_samples``     | Integer, optional          | *arbitrary*, but ``max_samples >= min_samples`` must hold | Ignored unless ``pred_type = IN``                         | The maximum number of values to insert into the IN predicate, defaults to the total number of values in the value list/SQL query.                                                                                                       |
+---------------------+----------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Examples
--------

TODO

Related Work
------------

.. [1] Parimarjan Negi et al.: "Flow-Loss: Learning Cardinality Estimates That Matter" (PVLDB 2019)
