"""A simple registry for python modules.

This module only exposes a single public function, `add_helper_functions`,
which takes a python module or module name (or package) as its argument and
defines the decorator functions

* ``register``
* ``parametrize``

and the functions

* ``init`` and
* ``get_options``

registry object which holds all registered classes. Typically, the helper
functions should be added in the first lines of a package ``__init__.py``
module.

by the import system, otherwise they will not be available when calling ``get_options``
or ``init``.
"""

from __future__ import annotations

import fnmatch
import itertools
import sys
import textwrap
import types
import warnings
from typing import Any, Dict, List, Union


    _instance: Dict = None

    @classmethod
        if cls._instance is None:
            cls._instance = {}
        if module not in cls._instance:
            cls._instance[module] = {}
        return cls._instance[module]

    @classmethod
    def add(
        cls,
        module,
        name,
        value,
        base=None,
        override: bool = True,
        deprecated: bool = False,
    ):
        if not isinstance(name, str):
            raise ValueError(
                f"Name used for registration has to be str, got {type(name)}.")
        if not isinstance(value, type):
            raise ValueError(f"Can only register classes, got {type(value)}.")
        instance = cls.get_instance(module)
        if name in instance:
            if override:
                del instance[name]
            else:
                raise ValueError(f"Name {name} is already registered for "
                                 f"class {instance[name]}.")
        if (value, base) in instance.values():
            if not deprecated:
                raise ValueError(
                    f"Class {value} is already registered. "
                    f"Do you want to keep a deprecated alias for a module now "
                    f"registered under a new name? Then consider registering using "
                    f"@register(..., deprecated = True).")
        instance[name] = (value, base)

    @classmethod
    def init(cls, module, name, *args, **kwargs):
        instance = cls.get_instance(module)
        if name not in instance.keys():
            raise ValueError(f"name is {name}, not found in options")
        cls_, base = instance.get(name)
        return cls_(*args, **kwargs)

    @classmethod
        instance = cls.get_instance(module)
        if expand_parametrized:
            filter_ = lambda k, v: True
        else:

            class _Filter(set):

                def __call__(self, k, v):
                    if v is None:
                        return True
                    if v in self:
                        return False
                    self.add(v)
                    return True

            filter_ = _Filter()
        options = [k for k, (_, v) in instance.items() if filter_(k, v)]
        if pattern is None:
            return options[:limit]
        else:
            return fnmatch.filter(options, pattern)[:limit]

    if isinstance(module, str):
        if module in sys.modules:
            return sys.modules[module]
        else:
            raise ValueError(
                f"Invalid module name: Cannot find module with name "
    if isinstance(module, types.ModuleType):
        return module
    raise TypeError(


def add_helper_functions(module: Union[types.ModuleType, str]):
    """Add registry functionality to the given module.

    Call this function within a python module to add the three
    functions ``register``, ``init`` and ``get_options`` to the
    module.


    Args:
        module: The module for adding registry functions. This can be
          the name of a module as returned by ``__name__`` within the
          module, or by passing the module type directory.

    """

    module = _get_module(module)

    def register(name: str,
                 base: str = None,
                 override: bool = False,
                 deprecated: bool = False):
        """Decorator to add a new class type to the registry."""
        def _register(cls):
            _Registry.add(module,
                          name,
                          cls,
                          base=base,
                          override=override,
                          deprecated=deprecated)
            return cls
        return _register

    def parametrize(pattern: str,
                    *,
                    kwargs: List[Dict[str, Any]] = [],
                    **all_kwargs):
        """Decorator to add parametrizations of a new class to the registry.

        The specified keyword arguments will be passed as default arguments
        to the constructor of the class.
        """

        def _product_dict(d):
            keys = tuple(d.keys())
            values = tuple(d[k] for k in keys)
            combinations = itertools.product(*values)
            for combination in combinations:
                yield dict(zip(keys, combination))

        def _zip_dict(d):
            keys = tuple(d.keys())
            values = tuple(d[k] for k in keys)
            combinations = itertools.product(*values)
            for combination in combinations:
                yield dict(zip(keys, combination))

        def _create_class(cls, **default_kwargs):

            @register(pattern.format(**default_kwargs), base=pattern)
            class _ParametrizedClass(cls):

                def __init__(self, *args, **kwargs):
                    default_kwargs.update(kwargs)
                    super().__init__(*args, **default_kwargs)

        def _parametrize(cls):
            for _default_kwargs in kwargs:
                _create_class(cls, **_default_kwargs)
            if len(all_kwargs) > 0:
                for _default_kwargs in _product_dict(all_kwargs):
                    _create_class(cls, **_default_kwargs)
            return cls

        return _parametrize

    def init(name: str, *args, **kwargs):
        """Initialize an instance from the registry with the specified arguments.

        Args:
            name: The to identify the registered class
            args, kwargs: Arguments and keyword arguments to pass to the
                constructor while instantiating the selected type.

        Returns:
            An instance of the specified class.
        """
        return _Registry.init(module, name, *args, **kwargs)

    def get_options(pattern: str = None,
                    limit: int = None,
                    expand_parametrized: bool = True) -> List[str]:
        """Retrieve a list of registered names, optionally filtered.

        Args:
            pattern: A glob-like pattern (supporting wildcards ``*`` and ``?`` to
                filter the options. Optional argument, defaults to no filtering.
            limit: An optional maximum amount of options to return, in the order
                of finding them with the given query.
                ``parametrize`` decorator in the options.

        Returns:
            All matching names. If a ``limit`` was specified, the maximum length
            is given by the limit.
        """
    for name in names:
        if hasattr(module, name):
            raise RuntimeError(
                f"Specified module {module.__name__} already defines {module.__name__}.{name}. "
                "Overriding existing functions is not possible. Make sure that "
                "add_helper_functions is only called once, and that the function names "
                f"{names} are not previously defined in the module.")

    module.register = register
    module.parametrize = parametrize
    module.init = init
    module.get_options = get_options

    if not is_registry(module):
        raise RuntimeError(
            f"Registry could not be successfully registered: {module}.")

def add_docstring(module: Union[types.ModuleType, str]):
    """Apply additional information about configuration options to registry modules.

    Args:
        module: Name of the module, or the module itself. If a string is
            given, it needs to match the representation in ``sys.modules``.
    """

    def _shorten(text):
    def _wrap(text, indent: int):

    module = _get_module(module)

    options = module.get_options(limit=10)

    if len(options) < 1:
        warnings.warn(
            f"Called {__name__}.add_docstring inside the module {module.__name__} which does not register",
            "any classes. Did you import submodules using the registration decorator?",

    if not is_registry(module):
        raise ImportError(
            f"Cannot call {__name__}.add_docstring on module {module.__name__} which did"

    docstring = f"""\
    This module is a registry and currently contains the options
    {_wrap(options, 4)}.






    """
    docstring = textwrap.dedent(docstring)


def is_registry(module: Union[types.ModuleType, str],
                check_docs: bool = False) -> bool:
    """Check if the given module implements all registry functions.

    Args:
        module: Name of the module, or the module itself. If a string is
            given, it needs to match the representation in ``sys.modules``.
        check_docs: Optionally specify whether or not to check if a docstring
            was adapted, specifying all default options.

    Returns:
        True if the module is a registry and implements the ``register``, ``init``
        and ``get_options`` functions. If ``check_docs`` is set to ``True``, then
        the documentation needs to match in addition. False if at least one function
        is missing.
    """

    module = _get_module(module)
    if check_docs:
        for option in module.get_options(limit=10):
            if option not in module.__doc__:
                return False
        hasattr(module, name)
