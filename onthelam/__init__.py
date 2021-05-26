import ast

OP_MAP = {
    '+': ast.Add(),
    '-': ast.Sub(),
}

class LambdaBuilder:
    def __init__(self, body=None, body_str=None, parent=None, closed_values=None, closed_count=0):
        if not body:
            body = ast.Name(id='_', ctx=ast.Load())
            body_str = '_'
        self.__body = body
        if not body_str:
            raise ValueError('body_str must be present if body is provided')
        self.__body_str = body_str
        self.__parent = parent
        if not closed_values:
            closed_values = []
        self.__closure = closed_values
        self.__closure_size = len(closed_values) + closed_count

    def __repr__(self):
        return '_ -> ' + self.__body_str

    def __op(self, op, other):
        new = LambdaBuilder(
            ast.BinOp(
                left=self.__body,
                op=OP_MAP[op],
                right=ast.Subscript(
                    value=ast.Name(id='closure', ctx=ast.Load()),
                    slice=ast.Constant(value=self.__closure_size),
                    ctx=ast.Load()
                )
            ),
            body_str=(f'{self.__body_str} {op} {repr(other)}'),
            parent=self,
            closed_values=[other],
            closed_count=self.__closure_size
        )
        return new

    def __rop(self, op, other):
        new = LambdaBuilder(
            ast.BinOp(
                left=ast.Subscript(
                    value=ast.Name(id='closure', ctx=ast.Load()),
                    slice=ast.Constant(value=self.__closure_size),
                    ctx=ast.Load()
                ),
                op=OP_MAP[op],
                right=self.__body,
            ),
            body_str=(f'{repr(other)} {op} {self.__body_str}'),
            parent=self,
            closed_values=[other],
            closed_count=self.__closure_size
        )
        return new

    def __add__(self, other):
        return self.__op('+', other)

    def __radd__(self, other):
        return self.__rop('+', other)

    def __sub__(self, other):
        return self.__op('-', other)

    def __rsub__(self, other):
        return self.__rop('-', other)

    def __call__(self, arg):
        fn = self.__compile()
        return fn(arg)

    def __compile(self):
        ast_object = ast.Expression(
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[ast.arg(arg='_')],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=self.__body,
            )
        )
        code = compile(
            ast.fix_missing_locations(ast_object),
            '<LambdaBuilder>',
            mode='eval'
        )
        return eval(code, {'closure': self.__assemble_closure()})

    def __assemble_closure(self):
        node = self
        closure = node.__closure
        while node.__parent:
            node = node.__parent
            closure[:0] = node.__closure
        if len(closure) != self.__closure_size:
            raise RuntimeError(
                f'Lambda closure corrupt. Expected size {self.__closure_size}'
                f' but found {len(closure)} stored values.', closure)
        return closure


_ = LambdaBuilder()
