#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import types

import pytest

import cebra.registry


def _make_module():
    test_module = types.ModuleType("registry_test",
                                   "Test module for initializing the registry")
    return test_module


def _make_class():

    class Foo:

        def __init__(self, x=42):
            self.x = x

    return Foo


def _make_registry():
    test_module = _make_module()
    cebra.registry.add_helper_functions(test_module)
    return test_module


def test_create_registry():
    test_module = _make_registry()
    assert cebra.registry.is_registry(test_module)


def test_register():
    test_module = _make_registry()
    Foo = _make_class()

    _Foo = test_module.register("foo")(Foo)
    assert _Foo == Foo, "registration should not alter the class."
    assert "foo" in test_module.get_options()

    instance = test_module.init("foo")
    assert isinstance(instance, Foo)


def test_get_options():
    test_module = _make_registry()

    @test_module.parametrize("foo-{bar}", bar=range(100))
    class Foo:

        def __init__(self, bar=None):
            pass

    # test that by default, there is no limit
    assert len(test_module.get_options()) == 100
    # test limit option
    assert len(test_module.get_options(limit=5)) == 5
    # test that output is sorted
    assert test_module.get_options(limit=2) == ["foo-0", "foo-1"]
    # test filtering
    assert len(test_module.get_options("foo-*0")) == 10
    assert test_module.get_options("foo-1*", limit=2) == ["foo-1", "foo-10"]


def test_double_registration_error():
    test_module = _make_registry()
    with pytest.raises(RuntimeError):
        cebra.registry.add_helper_functions(test_module)


def test_parametrize():
    test_module = _make_registry()
    Foo = _make_class()

    _Foo = test_module.parametrize("foo {x}", x=range(10))(Foo)
    assert _Foo == Foo, "registration should not alter the class."
    for i in range(10):
        assert f"foo {i}" in test_module.get_options()
    for i in range(10):
        instance = test_module.init(f"foo {i}")
        assert isinstance(instance, Foo)
        assert instance.x == i

    assert len(test_module.get_options(expand_parametrized=False)) == 1
    Bar = _make_class()
    test_module.register("bar")(Bar)
    assert len(test_module.get_options(expand_parametrized=False)) == 2


def test_override():
    test_module = _make_registry()
    Foo = _make_class()
    Bar = _make_class()
    _Foo1 = test_module.register("foo")(Foo)
    assert _Foo1 == Foo
    assert _Foo1 != Bar
    assert f"foo" in test_module.get_options()

    # Check that the class was actually added to the module
    assert (
        Foo,
        None) in cebra.registry._Registry.get_instance(test_module).values()

    # Using the same name raises an error
    with pytest.raises(ValueError):
        _ = test_module.register("foo")(Bar)

    # Registering the same class under different names
    # also raises and error*
    with pytest.raises(ValueError):
        _ = test_module.register("bar")(Foo)

    # Now it works
    _Foo2 = test_module.register("foo", override=True)(Bar)
    assert _Foo2 != Foo
    assert _Foo2 == Bar
    assert f"foo" in test_module.get_options()


def test_depreciation():
    test_module = _make_registry()
    Foo = _make_class()
    _Foo1 = test_module.register("foo")(Foo)
    assert _Foo1 == Foo
    assert f"foo" in test_module.get_options()

    # Registering the same class under different names
    # also raises and error
    with pytest.raises(ValueError):
        _ = test_module.register("bar")(Foo)

    # Unless you specify that the class is deprecated
    _ = test_module.register("bar", deprecated=True)(Foo)
    # but now using the module will raise a deprecation
    # warning when we initialize the class

    # TODO(stes): This requires a bigger change to how modules are
    # registered. Skipping for no and left as a future feature
    # to implement
    # with pytest.warns(DeprecationWarning):
    #    _ = test_module.init("bar")
