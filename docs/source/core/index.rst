Core concepts
=============

This section is dedicated to the core modules of PostBOUND and how they interact with each other.
If you are a new user, it is highly recommended to read through this section to get a better understanding of the framework.
At the same time, the :doc:`../10minutes` tutorial provides a more fast-paced introduction.

Contents
--------

.. toctree::
   :maxdepth: 2

   hinting
   qal
   databases
   optimization


Package structure
-----------------

===========================================  ==================================================================================================================================================================================================================================================================================================================================================================================
Package                                      Description
===========================================  ==================================================================================================================================================================================================================================================================================================================================================================================
*root*                                       The main entry point into PostBOUND as well as fundamental data structures like :class:`OptimizationPipeline <postbound.OptimizationPipeline>` or :class:`QueryPlan <postbound.optimizer.QueryPlan>` are located here. See :doc:`optimization` for details.
:mod:`optimizer <postbound.optimizer>`       Provides the optimizer-related data structures such as :class:`JoinGraph <postbound.optimizer.JoinGraph>`, :class:`JoinOrder <postbound.optimizer.JoinOrder>`, or :class:`PhysicalOperatorAssignment <postbound.optimizer.PhysicalOperatorAssignment>`, as well as :doc:`existing optimization algorithms <../advanced/existing-strategies>`. See :doc:`optimization` for details.
:mod:`qal <postbound.qal>`                   Home of the query abstraction used throughout PostBOUND, as well as logic to parse and transform query instances. See :doc:`qal` for details.
:mod:`db <postbound.db>`                     Contains all parts of PostBOUND that concern database interaction. That includes retrieving data from different database systems, as well as generating optimized queries to execute on the database system. Notably, the :class:`Database <postbound.db.Database>` interface is defined here. See :doc:`databases` for details.
:mod:`experiments <postbound.experiments>`   All tools to conveniently load benchmarks (:mod:`workloads <postbound.experiments.workloads>`) and to measure their execution time for different optimization settings are located here (from the :mod:`executor <postbound.experiments.executor>` module). See the package documentation for more details.
:mod:`util <postbound.util>`                 Utilities that are not really specific to query optimization find their home here. See the package documentation for more details.
:mod:`vis <postbound.vis>`                   Contains utilities to visualize different concepts in query optimization (join orders, join graphs, query execution plans, ...).
===========================================  ==================================================================================================================================================================================================================================================================================================================================================================================

On a high-level, these packages interact as follows:

.. figure:: ../../figures/postbound-package-overview.svg
   :align: center

   Interaction between the main PostBOUND packages.
