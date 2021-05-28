"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument.
"""

import ast
from copy import deepcopy
from enum import Enum
from dataclasses import dataclass, field, replace
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
from typing import (
    cast,
    Any,
    Callable,
    Iterator,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    NoReturn,
    Union,
)
from types import CodeType


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
        ast_op: Callable[..., ast.expr],
        operator: Union[UnaryFunction, BinaryFunction],
    ):
        self.precedence = precedence
        self.format_str = format_str
        self.ast_op = ast_op
        self.operator = operator

    def render(self, *args: Any) -> str:
        """Generate a string representation of an operation given its args"""
        return self.format_str.format(*args)


@dataclass(frozen=True)
class ClosureChain:
    """Maintains a chain of references to objects closed over in the lambda"""

    values: list[Any]
    previous: Optional["ClosureChain"] = None
    size: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "size",
            len(self.values) + (len(self.previous) if self.previous else 0),
        )

    def __iter__(self) -> Iterator[Any]:
        if self.previous:
            yield from self.previous
        yield from self.values

    def __len__(self) -> int:
        return self.size

    def chain(self, next_values: Iterable[Any]) -> "ClosureChain":
        """Add another chunk of values to the end of the chain"""
        return ClosureChain(list(next_values), self)

    @classmethod
    def new(cls) -> "ClosureChain":
        """Create a new empty chain"""
        return cls([])


@dataclass(frozen=True)
class Lambda:
    """Maintains the data model of a lambda function"""

    name: str
    tree: ast.expr
    closure: ClosureChain
    body_str: str
    last_op: Operation

    @classmethod
    def new(cls, arg_name: str) -> "Lambda":
        """Create an empty initial (identity) lambda"""
        return cls(
            arg_name, name(arg_name), ClosureChain.new(), arg_name, Operation.ARG
        )

    def __repr__(self) -> str:
        return f"{self.body_str}"

    def render(self) -> str:
        """Render the user-friendly lambda definition"""
        return f"{self.name} -> {self}"

    def compile(self) -> CodeType:
        """Create the code for a lambda with this object as its body"""
        lambda_ast = ast.Expression(
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[ast.arg(arg=self.name)],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=self.tree,
            )
        )
        lambda_ast = ast.fix_missing_locations(lambda_ast)
        code = compile(lambda_ast, "<LambdaBuilder>", mode="eval")
        # Unclear why Mypy thinks the "compile" function returns anything other
        # than a code object.
        return cast(CodeType, code)

    def merge_after(self, op: Operation, other: "Lambda") -> "Lambda":
        """Combine this Lambda with another according to the given operation."""

        other_ast = self.update_appended_closure_idxs(other.tree)
        return replace(
            self,
            tree=op.ast_op(self.tree, other_ast),
            closure=self.closure.chain(other.closure),
            body_str=op.render(self, other),
            last_op=op,
        )

    def update_appended_closure_idxs(self, tree: ast.expr) -> ast.expr:
        """Update the indexes of the given tree to start after this closure"""
        new_tree = deepcopy(tree)
        base = len(self.closure)
        for node in ast.walk(new_tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "closure"
                and isinstance(node.slice, ast.Constant)
            ):
                node.slice.value += base
        return new_tree

    def contextually_paren(self, op: Operation, from_right: bool = False) -> "Lambda":
        """Return this, but with parens around it if contextually warranted"""
        do_paren = self.last_op.precedence < op.precedence or (
            from_right and self.last_op.precedence == op.precedence
        )
        if do_paren and self.last_op is not Operation.PARENS:
            return replace(
                self,
                body_str=Operation.PARENS.render(self),
                last_op=Operation.PARENS,
            )
        return self


class LambdaBuilder:
    """Assembles the lambda by recording operations performed on it"""

    def __init__(
        self,
        start_from: Union[str, Lambda],
    ):
        if isinstance(start_from, str):
            self.__lambda = Lambda.new(start_from)
        else:
            self.__lambda = start_from

    def __repr__(self) -> str:
        return self.__lambda.render()

    def __op(
        self, op: Operation, other: Any, from_right: bool = False
    ) -> "LambdaBuilder":
        lambda_in_context = self.__lambda.contextually_paren(op, from_right)

        if isinstance(other, LambdaBuilder):
            # This situation means both sides are LambdaBuilder instances, and
            # since the left-side operations take precedence, from_right should
            # always be false here and doesn't need to be checked.

            # This cast is safe given the contract that it should only be called
            # for binary operations.
            operation = cast(BinaryFunction, op.operator)
            new = operation(
                lambda_in_context,
                other,
            )
            # Operations on a LambdaBuilder (except call) always return another
            # LambdaBuilder instance, so this is safe.
            return cast(LambdaBuilder, new)

        if isinstance(other, Lambda):
            # Unless someone is mucking about with Lambda instances directly
            # rather than just letting this class handle them (they shouldn't)
            # this situation arises directly from the immediately preceding
            # code, but now we are in the right-side LambdaBuilder instance.
            return LambdaBuilder(
                other.merge_after(op, lambda_in_context),
            )

        other_ast = ast.Subscript(
            value=ast.Name(id="closure", ctx=ast.Load()),
            slice=ast.Constant(value=len(self.__lambda.closure)),
            ctx=ast.Load(),
        )
        new_closure = self.__lambda.closure.chain([other])

        if from_right:
            new_tree = op.ast_op(other_ast, self.__lambda.tree)
            body_str = op.render(repr(other), lambda_in_context)
        else:
            new_tree = op.ast_op(self.__lambda.tree, other_ast)
            body_str = op.render(lambda_in_context, repr(other))

        return LambdaBuilder(
            replace(
                lambda_in_context,
                tree=new_tree,
                closure=new_closure,
                body_str=body_str,
                last_op=op,
            ),
        )

    def __uop(self, op: Operation) -> "LambdaBuilder":
        new = self.__lambda.contextually_paren(op, from_right=True)
        return LambdaBuilder(
            replace(
                new,
                tree=op.ast_op(self.__lambda.tree),
                body_str=op.render(new),
                last_op=op,
            ),
        )

    def __special_op(
        self, op: Union[Literal[Operation.GETATTR, Operation.GETITEM]], other: Any
    ) -> "LambdaBuilder":
        new = self.__lambda.contextually_paren(op)

        if op is Operation.GETATTR:
            body_str = op.render(new, other)
            new_tree = op.ast_op(self.__lambda.tree, str(other))
            new_closure = self.__lambda.closure
        elif op is Operation.GETITEM:
            other_ast = ast.Subscript(
                value=ast.Name(id="closure", ctx=ast.Load()),
                slice=ast.Constant(value=len(self.__lambda.closure)),
                ctx=ast.Load(),
            )
            body_str = op.render(new, repr(other))
            new_tree = op.ast_op(self.__lambda.tree, other_ast)
            new_closure = self.__lambda.closure.chain([other])

        return LambdaBuilder(
            replace(
                new,
                tree=new_tree,
                closure=new_closure,
                body_str=body_str,
                last_op=op,
            ),
        )

    def __bool__(self) -> NoReturn:
        raise NotImplementedError(
            "This object should not be directly tested for truthiness"
        )

    def __lt__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.LT, other)

    def __le__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.LE, other)

    def __eq__(self, other: Any) -> "LambdaBuilder":  # type: ignore
        return self.__op(Operation.EQ, other)

    def __ne__(self, other: Any) -> "LambdaBuilder":  # type: ignore
        return self.__op(Operation.NE, other)

    def __ge__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.GE, other)

    def __gt__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.GT, other)

    def __or__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.OR, other)

    def __ror__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.OR, other, from_right=True)

    def __xor__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.XOR, other)

    def __rxor__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.XOR, other, from_right=True)

    def __and__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.AND, other)

    def __rand__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.AND, other, from_right=True)

    def __lshift__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.LSHIFT, other)

    def __rlshift__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.LSHIFT, other, from_right=True)

    def __rshift__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.RSHIFT, other)

    def __rrshift__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.RSHIFT, other, from_right=True)

    def __add__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.ADD, other)

    def __radd__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.ADD, other, from_right=True)

    def __sub__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.SUB, other)

    def __rsub__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.SUB, other, from_right=True)

    def __mul__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MUL, other)

    def __rmul__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MUL, other, from_right=True)

    def __truediv__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.TRUEDIV, other)

    def __rtruediv__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.TRUEDIV, other, from_right=True)

    def __floordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.FLOORDIV, other)

    def __rfloordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.FLOORDIV, other, from_right=True)

    def __mod__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MOD, other)

    def __rmod__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MOD, other, from_right=True)

    def __matmul__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MATMUL, other)

    def __rmatmul__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MATMUL, other, from_right=True)

    def __pow__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.POW, other)

    def __rpow__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.POW, other, from_right=True)

    def __pos__(self) -> "LambdaBuilder":
        return self.__uop(Operation.POS)

    def __neg__(self) -> "LambdaBuilder":
        return self.__uop(Operation.NEG)

    def __invert__(self) -> "LambdaBuilder":
        return self.__uop(Operation.INVERT)

    def __getitem__(self, item: Any) -> "LambdaBuilder":
        return self.__special_op(Operation.GETITEM, item)

    def __getattr__(self, item: str) -> "LambdaBuilder":
        return self.__special_op(Operation.GETATTR, item)

    def __call__(self, arg: Any) -> Any:
        func = self.__compile()
        return func(arg)

    def __compile(self) -> Callable[[Any], Any]:
        code = self.__lambda.compile()
        closure = list(self.__lambda.closure)
        # Evaluating the AST we generate is key to the functioning of this
        # object, so we ignore the eval warning here.
        func = eval(code, {"closure": closure})  # pylint: disable=eval-used
        # This is guaranteed to be callable because the AST we are compiling
        # contains a single expression containing a single lambda accepting
        # one parameter.
        return cast(Callable[[Any], Any], func)


_ = LambdaBuilder("_")
