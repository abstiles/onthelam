import ast

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


class LambdaBuilder:
    def __init__(
        self,
        body=None,
        body_str=None,
        parent=None,
        closed_values=None,
        closed_count=0,
        precedence=100,
    ):
        if not body:
            body = ast.Name(id="_", ctx=ast.Load())
            body_str = "_"
        self.__body = body
        if not body_str:
            raise ValueError("body_str must be present if body is provided")
        self.__body_str = body_str
        self.__parent = parent
        if not closed_values:
            closed_values = []
        self.__closure = closed_values
        self.__closure_size = len(closed_values) + closed_count
        self.__precedence = precedence

    def __repr__(self):
        return "_ -> " + self.__body_str

    def __op(self, op, other):
        body_str = (
            f"({self.__body_str})"
            if self.__precedence < OP_PRECEDENCE[op]
            else self.__body_str
        )

        new = LambdaBuilder(
            ast.BinOp(
                left=self.__body,
                op=OP_MAP[op],
                right=ast.Subscript(
                    value=ast.Name(id="closure", ctx=ast.Load()),
                    slice=ast.Constant(value=self.__closure_size),
                    ctx=ast.Load(),
                ),
            ),
            body_str=(f"{body_str} {op} {repr(other)}"),
            parent=self,
            closed_values=[other],
            closed_count=self.__closure_size,
            precedence=OP_PRECEDENCE[op],
        )
        return new

    def __rop(self, op, other):
        body_str = (
            f"({self.__body_str})"
            if self.__precedence <= OP_PRECEDENCE[op]
            else self.__body_str
        )

        new = LambdaBuilder(
            ast.BinOp(
                left=ast.Subscript(
                    value=ast.Name(id="closure", ctx=ast.Load()),
                    slice=ast.Constant(value=self.__closure_size),
                    ctx=ast.Load(),
                ),
                op=OP_MAP[op],
                right=self.__body,
            ),
            body_str=(f"{repr(other)} {op} {body_str}"),
            parent=self,
            closed_values=[other],
            closed_count=self.__closure_size,
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
        fn = self.__compile()
        return fn(arg)

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
                body=self.__body,
            )
        )
        code = compile(
            ast.fix_missing_locations(ast_object), "<LambdaBuilder>", mode="eval"
        )
        return eval(code, {"closure": self.__assemble_closure()})

    def __assemble_closure(self):
        node = self
        closure = node.__closure
        while node.__parent:
            node = node.__parent
            closure[:0] = node.__closure
        if len(closure) != self.__closure_size:
            raise RuntimeError(
                f"Lambda closure corrupt. Expected size {self.__closure_size}"
                f" but found {len(closure)} stored values.",
                closure,
            )
        return closure


_ = LambdaBuilder()
