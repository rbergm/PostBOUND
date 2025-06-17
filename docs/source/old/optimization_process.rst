The optimization process in more detail
=======================================

This chapter highlights the different steps that need to be executed in order to optimize and execute an SQL query using
PostBOUND. It tries supply a mental model of the different steps to aid a tailored search for errors in case something goes
wrong. The entire optimization process is summarized in the below figure:

.. figure:: ../../figures/postbound-workflow.svg

    Query optimization process in PostBOUND.

The first step in every query optimization is to obtain an input query. Since PostBOUND uses an abstract model of SQL queries
(see :doc:`qal`), this query has to be parsed - either using the `parser` module, or by using an entire query workload via the
`workloads` module (which in turn delegates part of the work to the parser).

Once a usable input query exists, it is ready to be optimized. In PostBOUND the actual optimization happens within a
customizable pipeline. These pipelines are defined in the main module. Depending on the specific use-case, different
pipelines exist, for example to obtain a query execution plan in one integrated process (join ordering and physical operator
selection), or to run a multi-stage optimization process (with incremental improvements or join ordering and physical operator
selection in two separate phases). Once the pipeline that best fits the optimization scenario has been chosen, it has to be
initialized with actual optimization algorithms. These algorithms take care of pipeline-specific optimization steps, such as
finding the optimal join order. The general interfaces for each stage are defined algon with their pipelines in the main
module.

When an optimization pipeline terminates, the optimization decisions (e.g. cardinality estimates, join order or physical
operators) are encoded in an abstract format. Therefore, the last step in the optimization process is to ensure that these
decisions are actually enforced when the query is executed on an actual physical database system. However, PostBOUND does not
have a hook available that augments the optimizers of the database systems directly. Instead, an indirect approach is choosen.
The behavior of all the database systems supported by PostBOUND can be influenced via third-party extensions of standard SQL.
These extensions are typically called *hints* within the PostBOUND source code and documentation. These hints are either
proprietary extensions of the SQL syntax, or (more commonly) special comment blocks that can be inserted in certain places
within the query. The generation of appropriate hints is managed by the target database system, using functionality specified
by the `db` module. Each supported database system implements the central database interface defined in that module.
