"""Contains utilities that are not specific to PostBOUND's domain of databases and query optimization."""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(
    __name__,
    __file__,
)
