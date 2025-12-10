"""The optimizer package defines the central interfaces to implement optimization algorithms.

TODO: detailed documentation
"""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(
    __name__,
    __file__,
)
