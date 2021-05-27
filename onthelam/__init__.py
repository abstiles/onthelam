"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument.
"""

import ast
from collections import namedtuple
from dataclasses import dataclass, field, InitVar
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


@dataclass(frozen=True)
class LambdaBody:
    """The AST for a lambda and its string representation"""

    tree: ast.AST
    _str: str

    def __repr__(self) -> str:
        return self._str

    def compile(self, name) -> CodeType:
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

    def __op(self, op: str, other: Any) -> "LambdaBuilder":
        body_str = (
            f"({self.__body})"
            if self.__precedence < OP_PRECEDENCE[op]
            else str(self.__body)
        )

        new = LambdaBuilder(
            self.__name,
            LambdaBody(
                ast.BinOp(
                    left=self.__body.tree,
                    op=OP_MAP[op],
                    right=ast.Subscript(
                        value=ast.Name(id="closure", ctx=ast.Load()),
                        slice=ast.Constant(value=len(self.__closure)),
                        ctx=ast.Load(),
                    ),
                ),
                f"{body_str} {op} {repr(other)}",
            ),
            closure=self.__closure.chain([other]),
            precedence=OP_PRECEDENCE[op],
        )
        return new

    def __rop(self, op: str, other: Any) -> "LambdaBuilder":
        body_str = (
            f"({self.__body})"
            if self.__precedence <= OP_PRECEDENCE[op]
            else str(self.__body)
        )

        new = LambdaBuilder(
            self.__name,
            LambdaBody(
                ast.BinOp(
                    left=ast.Subscript(
                        value=ast.Name(id="closure", ctx=ast.Load()),
                        slice=ast.Constant(value=len(self.__closure)),
                        ctx=ast.Load(),
                    ),
                    op=OP_MAP[op],
                    right=self.__body.tree,
                ),
                f"{repr(other)} {op} {body_str}",
            ),
            closure=self.__closure.chain([other]),
            precedence=OP_PRECEDENCE[op],
        )
        return new

    def __add__(self, other: Any) -> "LambdaBuilder":
        return self.__op("+", other)

    def __radd__(self, other: Any) -> "LambdaBuilder":
        return self.__rop("+", other)

    def __sub__(self, other: Any) -> "LambdaBuilder":
        return self.__op("-", other)

    def __rsub__(self, other: Any) -> "LambdaBuilder":
        return self.__rop("-", other)

    def __mul__(self, other: Any) -> "LambdaBuilder":
        return self.__op("*", other)

    def __rmul__(self, other: Any) -> "LambdaBuilder":
        return self.__rop("*", other)

    def __truediv__(self, other: Any) -> "LambdaBuilder":
        return self.__op("/", other)

    def __rtruediv__(self, other: Any) -> "LambdaBuilder":
        return self.__rop("/", other)

    def __floordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__op("//", other)

    def __rfloordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__rop("//", other)

    def __mod__(self, other: Any) -> "LambdaBuilder":
        return self.__op("%", other)

    def __rmod__(self, other: Any) -> "LambdaBuilder":
        return self.__rop("%", other)

    def __matmul__(self, other: Any) -> "LambdaBuilder":
        return self.__op("@", other)

    def __rmatmul__(self, other: Any) -> "LambdaBuilder":
        return self.__rop("@", other)

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
