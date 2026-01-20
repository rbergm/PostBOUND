"""The optimizer module contains utilities to develop optimizers as well as simple optimization algorithms."""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(
    __name__,
    __file__,
)
