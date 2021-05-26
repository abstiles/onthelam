import ast
from unittest.mock import MagicMock

import pytest

from onthelam import LambdaBuilder, _


def test_simple_identity():
    fn = _
    assert repr(fn) == "_ -> _"
    assert fn(5) == 5


OBJ = MagicMock()


@pytest.mark.parametrize(
    ("expr", "body_str", "arg", "result"),
    [
        (lambda: _ + 5, "_ + 5", 2, 2 + 5),
        (lambda: 5 + _, "5 + _", 2, 5 + 2),
        (lambda: _ - 5, "_ - 5", 2, 2 - 5),
        (lambda: 5 - _, "5 - _", 2, 5 - 2),
        (lambda: _ * 2, "_ * 2", 3, 3 * 2),
        (lambda: 2 * _, "2 * _", 3, 2 * 3),
        (lambda: _ / 2, "_ / 2", 5, 5 / 2),
        (lambda: 2 / _, "2 / _", 5, 2 / 5),
        (lambda: _ // 2, "_ // 2", 5, 5 // 2),
        (lambda: 2 // _, "2 // _", 5, 2 // 5),
        (lambda: _ % 2, "_ % 2", 5, 5 % 2),
        (lambda: 2 % _, "2 % _", 5, 2 % 5),
        (lambda: _ @ 2, "_ @ 2", OBJ, OBJ @ 2),
        (lambda: 2 @ _, "2 @ _", OBJ, 2 @ OBJ),
    ],
)
def test_basic_ops(expr, body_str, arg, result):
    fn = expr()
    assert repr(fn) == f"_ -> {body_str}"
    assert fn(arg) == result


def test_missing_body_str():
    with pytest.raises(ValueError):
        fn = LambdaBuilder(body=ast.parse("_", mode="eval"))


def test_corrupt_closure():
    fn = LambdaBuilder(closed_count=1)
    with pytest.raises(RuntimeError):
        fn(5)


def test_add_multiple():
    fn = _ + ", hello" + " world"
    assert repr(fn) == "_ -> _ + ', hello' + ' world'"
    assert fn("hi") == "hi, hello world"


def test_add_both():
    fn = 1 + _ + 4
    assert repr(fn) == "_ -> 1 + _ + 4"
    assert fn(2) == 7


def test_arithmetic_precedence_no_parens():
    fn = 1 + _ * 2
    assert repr(fn) == "_ -> 1 + _ * 2"
    assert fn(3) == 7


def test_arithmetic_precedence_parens():
    fn = 1 + (_ + 2)
    assert repr(fn) == "_ -> 1 + (_ + 2)"
    assert fn(3) == 7


def test_arithmetic_precedence_parens():
    fn = (_ + 1) * 2
    assert repr(fn) == "_ -> (_ + 1) * 2"
    assert fn(3) == 8
