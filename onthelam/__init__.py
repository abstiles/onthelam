"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument.
"""

import ast
from enum import Enum
from dataclasses import dataclass, field
from typing import cast, Any, Callable, Iterator, NamedTuple, Optional, NoReturn
from types import CodeType


def compare(op: ast.operator, left: ast.expr, right: ast.expr) -> ast.Compare:
    """Generate an ast comparison"""
    return ast.Compare(left=left, ops=[op], comparators=[right])


def bin_op(op: ast.operator, left: ast.expr, right: ast.expr) -> ast.BinOp:
    """Generate an ast binary operation"""
    return ast.BinOp(left=left, op=op, right=right)


def unary_op(op: ast.operator, operand: ast.expr) -> ast.UnaryOp:
    """Generate an ast unary operation"""
    return ast.UnaryOp(op=op, operand=operand)


def subscript(value: ast.expr, idx: ast.expr) -> ast.Subscript:
    """Generate an ast subscript operation"""
    return ast.Subscript(value=value, slice=idx, ctx=ast.Load())


def attribute(value: ast.expr, attr: str) -> ast.Attribute:
    """Generate an ast attribute get operation"""
    return ast.Attribute(value=value, attr=attr, ctx=ast.Load())


NothingOperator = ast.operator()


# Precedence rules referenced from:
# https://docs.python.org/3/reference/expressions.html#operator-precedence
class Operation(Enum):
    """Represent an operation that can occur on an object"""

    LT = (5, "{} < {}", compare, ast.Lt())
    LE = (5, "{} <= {}", compare, ast.LtE())
    EQ = (5, "{} == {}", compare, ast.Eq())
    NE = (5, "{} != {}", compare, ast.NotEq())
    GE = (5, "{} >= {}", compare, ast.GtE())
    GT = (5, "{} > {}", compare, ast.Gt())
    OR = (6, "{} | {}", bin_op, ast.BitOr())
    XOR = (7, "{} ^ {}", bin_op, ast.BitXor())
    AND = (8, "{} & {}", bin_op, ast.BitAnd())
    LSHIFT = (9, "{} << {}", bin_op, ast.LShift())
    RSHIFT = (9, "{} >> {}", bin_op, ast.RShift())
    ADD = (10, "{} + {}", bin_op, ast.Add())
    SUB = (10, "{} - {}", bin_op, ast.Sub())
    MUL = (11, "{} * {}", bin_op, ast.Mult())
    MATMUL = (11, "{} @ {}", bin_op, ast.MatMult())
    TRUEDIV = (11, "{} / {}", bin_op, ast.Div())
    FLOORDIV = (11, "{} // {}", bin_op, ast.FloorDiv())
    MOD = (11, "{} % {}", bin_op, ast.Mod())
    NEG = (12, "-{}", unary_op, ast.USub())
    POS = (12, "+{}", unary_op, ast.UAdd())
    INVERT = (12, "~{}", unary_op, ast.Invert())
    POW = (13, "{} ** {}", bin_op, ast.Pow())

    # The folowing are not considered BinaryOps and require special handling.
    GETITEM = (15, "{}[{}]", subscript, NothingOperator)
    GETATTR = (15, "{}.{}", subscript, NothingOperator)

    def __init__(
        self,
        precedence: int,
        format_str: str,
        operation: Callable[..., ast.expr],
        ast_op: ast.operator,
    ):
        self.precedence = precedence
        self.format_str = format_str
        self.operation = operation
        self.ast_op = ast_op

    def format(self, *args: str) -> str:
        """Generate a string representation of an operation given its args"""
        return self.format_str.format(*args)


@dataclass(frozen=True)
class LambdaBody:
    """The AST for a lambda and its string representation"""

    tree: ast.expr
    _str: str

    def __repr__(self) -> str:
        return self._str

    def compile(self, name: str) -> CodeType:
        """Create the code for a lambda with this object as its body"""
        lambda_ast = ast.Expression(
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[ast.arg(arg=name)],
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

    def chain(self, next_values: list[Any]) -> "ClosureChain":
        """Add another chunk of values to the end of the chain"""
        return ClosureChain(next_values, self)

    @classmethod
    def new(cls) -> "ClosureChain":
        """Create a new empty chain"""
        return cls([])


class LambdaBuilder:
    """Assembles the lambda by recording operations performed on it"""

    def __init__(
        self,
        name: str,
        body: Optional[LambdaBody] = None,
        closure: ClosureChain = ClosureChain.new(),
        precedence: int = 100,
    ):
        if not name.isidentifier():
            raise ValueError("name must be a valid Python identifier")
        self.__name = name
        if not body:
            body = LambdaBody(ast.Name(id=name, ctx=ast.Load()), name)
        self.__body = body
        self.__closure = closure
        self.__precedence = precedence

    def __repr__(self) -> str:
        return f"{self.__name} -> {self.__body}"

    def __op(
        self, op: Operation, other: Any, from_right: bool = False
    ) -> "LambdaBuilder":
        do_paren = self.__precedence < op.precedence or (
            from_right and self.__precedence == op.precedence
        )
        body_str = f"({self.__body})" if do_paren else str(self.__body)

        other_ast = ast.Subscript(
            value=ast.Name(id="closure", ctx=ast.Load()),
            slice=ast.Constant(value=len(self.__closure)),
            ctx=ast.Load(),
        )

        if from_right:
            new_body = LambdaBody(
                op.operation(op.ast_op, other_ast, self.__body.tree),
                op.format(repr(other), body_str),
            )
        else:
            new_body = LambdaBody(
                op.operation(op.ast_op, self.__body.tree, other_ast),
                op.format(body_str, repr(other)),
            )

        return LambdaBuilder(
            self.__name,
            new_body,
            closure=self.__closure.chain([other]),
            precedence=op.precedence,
        )

    def __uop(self, op: Operation) -> "LambdaBuilder":
        do_paren = self.__precedence <= op.precedence
        body_str = f"({self.__body})" if do_paren else str(self.__body)

        return LambdaBuilder(
            self.__name,
            LambdaBody(
                unary_op(op.ast_op, self.__body.tree),
                op.format(body_str),
            ),
            closure=self.__closure,
            precedence=op.precedence,
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

    def __call__(self, arg: Any) -> Any:
        func = self.__compile()
        return func(arg)

    def __compile(self) -> Callable[[Any], Any]:
        code = self.__body.compile(self.__name)
        closure = list(self.__closure)
        # Evaluating the AST we generate is key to the functioning of this
        # object, so we ignore the eval warning here.
        func = eval(code, {"closure": closure})  # pylint: disable=eval-used
        # This is guaranteed to be callable because the AST we are compiling
        # contains a single expression containing a single lambda accepting
        # one parameter.
        return cast(Callable[[Any], Any], func)


_ = LambdaBuilder("_")
