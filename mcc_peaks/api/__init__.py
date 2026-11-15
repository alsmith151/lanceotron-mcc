"""Peak-calling APIs, loaded on demand to keep NumPy-only modes lightweight."""

from importlib import import_module

__all__ = [
    "io",
    "adaptive",
    "background",
    "counts",
    "junctions",
    "junction_plotting",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
