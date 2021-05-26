import ast

import pytest

from onthelam import LambdaBuilder, _

def test_simple_identity():
    fn = _
    assert repr(fn) == '_ -> _'
    assert fn(5) == 5


def test_missing_body_str():
    with pytest.raises(ValueError):
        fn = LambdaBuilder(body=ast.parse('_', mode='eval'))


def test_add():
    fn = _ + 5
    assert repr(fn) == '_ -> _ + 5'
    assert fn(2) == 7


def test_add_multiple():
    fn = _ + ', hello' + ' world'
    assert repr(fn) == "_ -> _ + ', hello' + ' world'"
    assert fn('hi') == 'hi, hello world'


def test_corrupt_closure():
    fn = LambdaBuilder(closed_count=1)
    with pytest.raises(RuntimeError):
        fn(5)


def test_radd():
    fn = 5 + _
    assert repr(fn) == '_ -> 5 + _'
    assert fn(2) == 7


def test_add_both():
    fn = 1 + _ + 4
    assert repr(fn) == '_ -> 1 + _ + 4'
    assert fn(2) == 7


def test_sub():
    fn = _ - 5
    assert repr(fn) == '_ -> _ - 5'
    assert fn(2) == -3


def test_rsub():
    fn = 5 - _
    assert repr(fn) == '_ -> 5 - _'
    assert fn(2) == 3
