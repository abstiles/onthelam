"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument.
"""

import ast
from collections import namedtuple
from dataclasses import dataclass, field

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


LambdaBody = namedtuple("LambdaBody", ["tree", "code_str"])


@dataclass(frozen=True)
class ClosureChain:
    """Maintains a chain of references to objects closed over in the lambda"""

    values: list
    previous: "ClosureChain" = None
    size: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "size",
            len(self.values) + (len(self.previous) if self.previous else 0),
        )

    def __iter__(self):
        if self.previous:
            yield from self.previous
        yield from self.values

    def __len__(self):
        return self.size

    def chain(self, next_values):
        """Add another chunk of values to the end of the chain"""
        return ClosureChain(next_values, self)

    @classmethod
    def new(cls):
        """Create a new empty chain"""
        return cls([])


class LambdaBuilder:
    """Assembles the lambda by recording operations performed on it"""

    def __init__(
        self,
        body=None,
        closure=ClosureChain.new(),
        precedence=100,
    ):
        if not body:
            body = LambdaBody(ast.Name(id="_", ctx=ast.Load()), code_str="_")
        self.__body = body
        self.__closure = closure
        self.__precedence = precedence

    def __repr__(self):
        return "_ -> " + self.__body.code_str

    def __op(self, op, other):
        body_str = (
            f"({self.__body.code_str})"
            if self.__precedence < OP_PRECEDENCE[op]
            else self.__body.code_str
        )

        new = LambdaBuilder(
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
                code_str=(f"{body_str} {op} {repr(other)}"),
            ),
            closure=self.__closure.chain([other]),
            precedence=OP_PRECEDENCE[op],
        )
        return new

    def __rop(self, op, other):
        body_str = (
            f"({self.__body.code_str})"
            if self.__precedence <= OP_PRECEDENCE[op]
            else self.__body.code_str
        )

        new = LambdaBuilder(
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
                code_str=(f"{repr(other)} {op} {body_str}"),
            ),
            closure=self.__closure.chain([other]),
            precedence=OP_PRECEDENCE[op],
        )
        return new

    def __add__(self, other):
        return self.__op("+", other)

    def __radd__(self, other):
        return self.__rop("+", other)

    def __sub__(self, other):
        return self.__op("-", other)

    def __rsub__(self, other):
        return self.__rop("-", other)

    def __mul__(self, other):
        return self.__op("*", other)

    def __rmul__(self, other):
        return self.__rop("*", other)

    def __truediv__(self, other):
        return self.__op("/", other)

    def __rtruediv__(self, other):
        return self.__rop("/", other)

    def __floordiv__(self, other):
        return self.__op("//", other)

    def __rfloordiv__(self, other):
        return self.__rop("//", other)

    def __mod__(self, other):
        return self.__op("%", other)

    def __rmod__(self, other):
        return self.__rop("%", other)

    def __matmul__(self, other):
        return self.__op("@", other)

    def __rmatmul__(self, other):
        return self.__rop("@", other)

    def __call__(self, arg):
        func = self.__compile()
        return func(arg)

    def __compile(self):
        ast_object = ast.Expression(
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[ast.arg(arg="_")],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=self.__body.tree,
            )
        )
        code = compile(
            ast.fix_missing_locations(ast_object), "<LambdaBuilder>", mode="eval"
        )
        closure = list(self.__closure)
        # Evaluating the AST we generate is key to the functioning of this
        # object, so we ignore the eval warning here.
        return eval(code, {"closure": closure})  # pylint: disable=eval-used


_ = LambdaBuilder()
