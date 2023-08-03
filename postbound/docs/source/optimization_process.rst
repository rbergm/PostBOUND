The optimization process in more detail
=======================================

This part of the documentation is still work in progress. In the future, it will describe the entire optimization process in
greater detail. More specifically, it will focus on the optimization pipeline, the different optimization stages and
interaction with database systems. For now, please refer to the API documentation of the ``postbound`` module, the ``stages``
module and the ``postbound.db`` package.

The planned high level scenario will start at parsing input queries (``postbound.qal`` package) to join ordering and physical
operator selection (``postbound`` and ``stages`` modules) to hinting (``db`` module).
