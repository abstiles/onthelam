from unittest.mock import MagicMock

import pytest

from onthelam.lambdabuilder import LambdaBuilder, Arg


_ = LambdaBuilder("_")
OBJ = MagicMock()


def test_simple_identity():
    fn = _
    assert repr(fn) == "_ -> _"
    assert fn(5) == 5


def test_rename():
    fn = LambdaBuilder("x")
    assert repr(fn) == "x -> x"
    assert fn(5) == 5


def test_invalid_name():
    with pytest.raises(ValueError):
        LambdaBuilder("123")


def test_bools_bad():
    with pytest.raises(NotImplementedError):
        bool(_)


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
        (lambda: _ == 2, "_ == 2", 2, 2 == 2),
        (lambda: _ == 2, "_ == 2", 1, 1 == 2),
        (lambda: _ != 2, "_ != 2", 2, 2 != 2),
        (lambda: _ != 2, "_ != 2", 1, 1 != 2),
        (lambda: _ < 2, "_ < 2", 1, 1 < 2),
        (lambda: _ < 2, "_ < 2", 2, 2 < 2),
        (lambda: _ <= 2, "_ <= 2", 2, 2 <= 2),
        (lambda: _ <= 2, "_ <= 2", 3, 3 <= 2),
        (lambda: _ > 2, "_ > 2", 1, 1 > 2),
        (lambda: _ > 2, "_ > 2", 3, 3 > 2),
        (lambda: _ >= 2, "_ >= 2", 1, 1 >= 2),
        (lambda: _ >= 2, "_ >= 2", 2, 2 >= 2),
        # 12 and 10 chosen to maximize bit pattern combinations.
        (lambda: _ | 12, "_ | 12", 10, 0b1010 | 0b1100),
        (lambda: 12 | _, "12 | _", 10, 0b1100 | 0b1010),
        (lambda: _ & 12, "_ & 12", 10, 0b1010 & 0b1100),
        (lambda: 12 & _, "12 & _", 10, 0b1100 & 0b1010),
        (lambda: _ ^ 12, "_ ^ 12", 10, 0b1010 ^ 0b1100),
        (lambda: 12 ^ _, "12 ^ _", 10, 0b1100 ^ 0b1010),
        (lambda: _ << 2, "_ << 2", 3, 3 << 2),
        (lambda: 2 << _, "2 << _", 3, 2 << 3),
        (lambda: _ >> 2, "_ >> 2", 12, 12 >> 2),
        (lambda: 2 >> _, "2 >> _", 1, 2 >> 1),
        (lambda: _ ** 2, "_ ** 2", 3, 3 ** 2),
        (lambda: 2 ** _, "2 ** _", 3, 2 ** 3),
        (lambda: +_, "+_", OBJ, +OBJ),
        (lambda: -_, "-_", 3, -3),
        (lambda: ~_, "~_", 5, ~5),
        (lambda: _[0], "_[0]", [1, 2], [1, 2][0]),
        (lambda: _.attr, "_.attr", OBJ, OBJ.attr),
    ],
)
def test_basic_ops(expr, body_str, arg, result):
    fn = expr()
    assert repr(fn) == f"_ -> {body_str}"
    assert fn(arg) == result


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


def test_unary_precedence_parens():
    fn = -(-_)
    assert repr(fn) == "_ -> -(-_)"
    assert fn(2) == 2


def test_chained_binary_op_and_compare():
    assert list(filter(_ % 2 == 0, range(10))) == list(range(0, 10, 2))


def test_repeated():
    fn = _ + _
    assert repr(fn) == "_ -> _ + _"
    assert fn(3) == 6


def test_repeated_with_merged_closures():
    fn = (_ + 2) * (_ + 3)
    assert repr(fn) == "_ -> (_ + 2) * (_ + 3)"
    assert fn(3) == 30


def test_multi_args():
    _1 = LambdaBuilder("_1")
    _2 = LambdaBuilder("_2")
    fn = _1 + _2
    assert repr(fn) == "_1, _2 -> _1 + _2"
    assert fn(1, 2) == 3


def test_multi_args_with_getattr():
    _1 = LambdaBuilder("_1")
    _2 = LambdaBuilder("_2")
    fn = _1[_2]
    assert repr(fn) == "_1, _2 -> _1[_2]"
    assert fn([1, 2, 3], 2) == 3


def test_multi_keyword_args():
    _1 = LambdaBuilder("_1")
    _2 = LambdaBuilder("_2")
    fn = _1[_2]
    assert repr(fn) == "_1, _2 -> _1[_2]"
    assert fn([1, 2, 3], 2) == 3
    assert fn(_2=2, _1=[1, 2, 3]) == 3


def test_arg_sort_by_name():
    _1 = LambdaBuilder("_1")
    _2 = LambdaBuilder("_2")
    fn = _2[_1]
    assert repr(fn) == "_1, _2 -> _2[_1]"
    assert fn(2, [1, 2, 3]) == 3


def test_arg_1():
    args = Arg("foo")
    assert len(args) == 1
    assert list(args) == ["foo"]


def test_args_multi():
    args = Arg("foo")
    args = args.merge(Arg("bar"))
    assert len(args) == 2


def test_args_multi_sort():
    args = Arg("foo")
    args = args.merge(Arg("bar"))
    assert list(args) == ["bar", "foo"]


def test_args_bad_merge():
    args = Arg("foo")
    with pytest.raises(TypeError):
        args.merge(["these", "args", "are", "not", "in", "order"])
