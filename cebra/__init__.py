"""CEBRA is a library for estimating Consistent Embeddings of high-dimensional Recordings 
using Auxiliary variables. It contains self-supervised learning algorithms implemented in
PyTorch, and has support for a variety of different datasets common in biology and neuroscience.
"""

is_sklearn_available = False
try:
    # TODO(stes): More common integrations people care about (e.g. PyTorch lightning)
    # could be added here.
    is_sklearn_available = True
except ImportError as e:
    # silently fail for now
    pass

__version__ = "0.1.0"
__all__ = ["CEBRA"]
__allow_lazy_imports = False
__lazy_imports = {}


def allow_lazy_imports():
    """Enables lazy imports of all submodules and packages of cebra.

    If called, references to ``cebra.<module_name>`` will be automatically
    lazily imported when first called in the code, and not raise a warning.
    """
    __allow_lazy_imports = True


def __getattr__(key):
    """Lazy import of cebra submodules and -packages.


    """
    if key == "CEBRA":
        return CEBRA
    elif not key.startswith("_"):
        import importlib
        import warnings
        if key not in __lazy_imports:
            if not __allow_lazy_imports:
                warnings.warn(
                    f"Your code triggered a lazy import of {__name__}.{key}. "
                    f"While this will (likely) work, it is recommended to "
                    f"add an explicit import statement to you code instead. "
                    f"To disable this warning, you can run "
                    f"``cebra.allow_lazy_imports()``.")
        return __lazy_imports[key]
    raise AttributeError(f"module 'cebra' has no attribute '{key}'. "
                         f"Did you import cebra.{key}?")
