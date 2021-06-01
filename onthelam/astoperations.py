"""
Models and functions for interacting with AST operations.
"""

import ast
from enum import Enum
from operator import (
    add,
    and_,
    eq,
    floordiv,
    ge,
    getitem,
    gt,
    invert,
    le,
    lshift,
    lt,
    matmul,
    mod,
    mul,
    ne,
    neg,
    or_,
    pos,
    rshift,
    sub,
    truediv,
    xor,
)
from typing import Any, Callable, Iterable, Union


def compare(op: ast.cmpop) -> Callable[[ast.expr, ast.expr], ast.Compare]:
    """Generate an ast comparison"""

    def _compare(left: ast.expr, right: ast.expr) -> ast.Compare:
        return ast.Compare(left=left, ops=[op], comparators=[right])

    return _compare


def bin_op(op: ast.operator) -> Callable[[ast.expr, ast.expr], ast.BinOp]:
    """Generate an ast binary operation"""

    def _bin_op(left: ast.expr, right: ast.expr) -> ast.BinOp:
        return ast.BinOp(left=left, op=op, right=right)

    return _bin_op


def unary_op(op: ast.unaryop) -> Callable[[ast.expr], ast.UnaryOp]:
    """Generate an ast unary operation"""

    def _unary_op(operand: ast.expr) -> ast.UnaryOp:
        return ast.UnaryOp(op=op, operand=operand)

    return _unary_op


def subscript(value: ast.expr, idx: ast.expr) -> ast.Subscript:
    """Generate an ast subscript operation"""
    return ast.Subscript(value=value, slice=idx, ctx=ast.Load())


def attribute(value: ast.expr, attr: str) -> ast.Attribute:
    """Generate an ast attribute get operation"""
    return ast.Attribute(value=value, attr=attr, ctx=ast.Load())


def name(value: str) -> ast.Name:
    """Generate an ast name expression"""
    if not value.isidentifier():
        raise ValueError("name must be a valid Python identifier")
    return ast.Name(id=value, ctx=ast.Load())


def constant(value: Any) -> ast.Constant:
    """Generate an ast constant expression"""
    return ast.Constant(value=value)


def lambda_expression(args: Iterable[str], tree: ast.expr) -> ast.Expression:
    """Generate an ast Expression defining a Lambda"""
    lambda_ast = ast.Expression(
        ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[*map(ast.arg, args)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=tree,
        )
    )
    return ast.fix_missing_locations(lambda_ast)


UnaryFunction = Callable[[Any], Any]
BinaryFunction = Callable[[Any, Any], Any]


# Precedence rules referenced from:
# https://docs.python.org/3/reference/expressions.html#operator-precedence
class Operation(Enum):
    """Represent an operation that can occur on an object"""

    LT = (5, "{} < {}", compare(ast.Lt()), lt)
    LE = (5, "{} <= {}", compare(ast.LtE()), le)
    EQ = (5, "{} == {}", compare(ast.Eq()), eq)
    NE = (5, "{} != {}", compare(ast.NotEq()), ne)
    GE = (5, "{} >= {}", compare(ast.GtE()), ge)
    GT = (5, "{} > {}", compare(ast.Gt()), gt)
    OR = (6, "{} | {}", bin_op(ast.BitOr()), or_)
    XOR = (7, "{} ^ {}", bin_op(ast.BitXor()), xor)
    AND = (8, "{} & {}", bin_op(ast.BitAnd()), and_)
    LSHIFT = (9, "{} << {}", bin_op(ast.LShift()), lshift)
    RSHIFT = (9, "{} >> {}", bin_op(ast.RShift()), rshift)
    ADD = (10, "{} + {}", bin_op(ast.Add()), add)
    SUB = (10, "{} - {}", bin_op(ast.Sub()), sub)
    MUL = (11, "{} * {}", bin_op(ast.Mult()), mul)
    MATMUL = (11, "{} @ {}", bin_op(ast.MatMult()), matmul)
    TRUEDIV = (11, "{} / {}", bin_op(ast.Div()), truediv)
    FLOORDIV = (11, "{} // {}", bin_op(ast.FloorDiv()), floordiv)
    MOD = (11, "{} % {}", bin_op(ast.Mod()), mod)
    NEG = (12, "-{}", unary_op(ast.USub()), neg)
    POS = (12, "+{}", unary_op(ast.UAdd()), pos)
    INVERT = (12, "~{}", unary_op(ast.Invert()), invert)
    POW = (13, "{} ** {}", bin_op(ast.Pow()), pow)

    # The folowing are not considered BinaryOps and require special handling.
    GETITEM = (15, "{}[{}]", subscript, getitem)
    GETATTR = (15, "{}.{}", attribute, getattr)

    # Special no-op operation only affecting the user-friendly string.
    PARENS = (16, "({})", lambda x: x, lambda x: x)

    # Special no-op "identity" operation representing the argument itself.
    ARG = (100, "{}", name, lambda x: x)

    def __init__(
        self,
        precedence: int,
        format_str: str,
        ast_renderer: Callable[..., ast.expr],
        operation_function: Union[UnaryFunction, BinaryFunction],
    ):
        self._precedence = precedence
        self._format_str = format_str
        self._ast_renderer = ast_renderer
        self._operation_function = operation_function

    @property
    def render_ast(self) -> Callable[..., ast.expr]:
        """Generate the AST expression object for this operation on its args"""
        return self._ast_renderer

    def render_str(self, *args: Any) -> str:
        """Generate a string representation of an operation given its args"""
        return self._format_str.format(*args)

    def __call__(self, *args: Any) -> Any:
        return self._operation_function(*args)

    def __ge__(self, other: "Operation") -> bool:
        if self.__class__ is other.__class__:
            return self._precedence >= other._precedence
        return NotImplemented

    def __gt__(self, other: "Operation") -> bool:
        if self.__class__ is other.__class__:
            return self._precedence > other._precedence
        return NotImplemented

    def __le__(self, other: "Operation") -> bool:
        if self.__class__ is other.__class__:
            return self._precedence <= other._precedence
        return NotImplemented

    def __lt__(self, other: "Operation") -> bool:
        if self.__class__ is other.__class__:
            return self._precedence < other._precedence
        return NotImplemented
