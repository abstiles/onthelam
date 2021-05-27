"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument.
"""

import ast
import operator
from enum import Enum
from dataclasses import dataclass, field
from typing import cast, Any, Callable, Iterator, NamedTuple, Optional
from types import CodeType

OP_MAP = {
    "+": ast.Add(),
    "-": ast.Sub(),
    "*": ast.Mult(),
    "@": ast.MatMult(),
    "/": ast.Div(),
    "//": ast.FloorDiv(),
    "%": ast.Mod(),
}

# Referenced from https://docs.python.org/3/reference/expressions.html#operator-precedence
OP_PRECEDENCE = {
    "bool": 1,
    "or": 2,
    "and": 3,
    "not": 4,
    "in": 5,
    "is": 5,
    "<": 5,
    "<=": 5,
    "==": 5,
    "!=": 5,
    ">=": 5,
    ">": 5,
    "|": 6,
    "^": 7,
    "&": 8,
    "<<": 9,
    ">>": 9,
    "+": 10,
    "-": 10,
    "*": 11,
    "@": 11,
    "/": 11,
    "//": 11,
    "%": 11,
    "+x": 12,
    "-x": 12,
    "~x": 12,
    "**": 13,
    "await": 14,
    "[]": 15,
    ".": 15,
    "x()": 15,
}


class Operation(Enum):
    """Represent an operation that can occur on an object"""

    ADD = (10, "{} + {}", operator.add, ast.Add())
    SUB = (10, "{} - {}", operator.sub, ast.Sub())
    MUL = (11, "{} * {}", operator.mul, ast.Mult())
    MATMUL = (11, "{} @ {}", operator.matmul, ast.MatMult())
    TRUEDIV = (11, "{} / {}", operator.truediv, ast.Div())
    FLOORDIV = (11, "{} // {}", operator.floordiv, ast.FloorDiv())
    MOD = (11, "{} % {}", operator.mod, ast.Mod())

    def __init__(
        self,
        precedence: int,
        format_str: str,
        operation: Callable[..., Any],
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

    tree: ast.AST
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

    def __op(self, op: Operation, other: Any) -> "LambdaBuilder":
        body_str = (
            f"({self.__body})"
            if self.__precedence < op.precedence
            else str(self.__body)
        )

        right = ast.Subscript(
            value=ast.Name(id="closure", ctx=ast.Load()),
            slice=ast.Constant(value=len(self.__closure)),
            ctx=ast.Load(),
        )

        new = LambdaBuilder(
            self.__name,
            LambdaBody(
                ast.BinOp(
                    left=self.__body.tree,
                    op=op.ast_op,
                    right=right,
                ),
                op.format(body_str, repr(other)),
            ),
            closure=self.__closure.chain([other]),
            precedence=op.precedence,
        )
        return new

    def __rop(self, op: Operation, other: Any) -> "LambdaBuilder":
        body_str = (
            f"({self.__body})"
            if self.__precedence <= op.precedence
            else str(self.__body)
        )

        left = ast.Subscript(
            value=ast.Name(id="closure", ctx=ast.Load()),
            slice=ast.Constant(value=len(self.__closure)),
            ctx=ast.Load(),
        )
        new = LambdaBuilder(
            self.__name,
            LambdaBody(
                ast.BinOp(left=left, op=op.ast_op, right=self.__body.tree),
                op.format(repr(other), body_str),
            ),
            closure=self.__closure.chain([other]),
            precedence=op.precedence,
        )
        return new

    def __add__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.ADD, other)

    def __radd__(self, other: Any) -> "LambdaBuilder":
        return self.__rop(Operation.ADD, other)

    def __sub__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.SUB, other)

    def __rsub__(self, other: Any) -> "LambdaBuilder":
        return self.__rop(Operation.SUB, other)

    def __mul__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MUL, other)

    def __rmul__(self, other: Any) -> "LambdaBuilder":
        return self.__rop(Operation.MUL, other)

    def __truediv__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.TRUEDIV, other)

    def __rtruediv__(self, other: Any) -> "LambdaBuilder":
        return self.__rop(Operation.TRUEDIV, other)

    def __floordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.FLOORDIV, other)

    def __rfloordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__rop(Operation.FLOORDIV, other)

    def __mod__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MOD, other)

    def __rmod__(self, other: Any) -> "LambdaBuilder":
        return self.__rop(Operation.MOD, other)

    def __matmul__(self, other: Any) -> "LambdaBuilder":
        return self.__op(Operation.MATMUL, other)

    def __rmatmul__(self, other: Any) -> "LambdaBuilder":
        return self.__rop(Operation.MATMUL, other)

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
