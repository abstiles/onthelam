"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument.
"""

import ast
import heapq
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import groupby
from operator import itemgetter
from typing import (
    cast,
    Any,
    Callable,
    Iterator,
    Iterable,
    Literal,
    Optional,
    NoReturn,
    TypeVar,
    Union,
)
from types import CodeType

from .astoperations import Operation, name, subscript, constant, lambda_expression


T = TypeVar("T")


class ArgList:
    """The list of args for a lambda

    Do not instantiate directly. Build by merging existing Arg and ArgList
    instances.

    """

    def __init__(self, arg_list: Iterable[str]):
        # Possibly redundant sorting is cheap with Python's sorting algorithm.
        self._arg_list = sorted(arg_list)

    def merge(self, other: "ArgList") -> "ArgList":
        """Combine two ArgLists, preserving sort order"""
        if not isinstance(other, ArgList):
            raise TypeError("Attempt to merge ArgList with non-ArgList instance")
        return ArgList(uniq(heapq.merge(self, other)))

    def __iter__(self) -> Iterator[str]:
        return iter(self._arg_list)

    def __len__(self) -> int:
        return len(self._arg_list)


class Arg(ArgList):
    """Helper constructor to simplify the single-arg case"""

    def __init__(self, identifier: str):
        if not identifier.isidentifier():
            raise ValueError("name must be a valid Python identifier")
        super().__init__([identifier])

    def __len__(self) -> Literal[1]:
        return 1


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

    args: ArgList
    tree: ast.expr
    closure: ClosureChain
    body_str: str
    last_op: Operation

    @classmethod
    def new(cls, arg_name: str) -> "Lambda":
        """Create an empty initial (identity) lambda"""
        return cls(
            Arg(arg_name), name(arg_name), ClosureChain.new(), arg_name, Operation.ARG
        )

    def __repr__(self) -> str:
        return f"{self.body_str}"

    def render(self) -> str:
        """Render the user-friendly lambda definition"""
        return f"{', '.join(self.args)} -> {self}"

    def compile(self) -> CodeType:
        """Create the code for a lambda with this object as its body"""
        code = compile(
            lambda_expression(iter(self.args), self.tree),
            "<LambdaBuilder>",
            mode="eval",
        )
        # Unclear why Mypy thinks the "compile" function returns anything other
        # than a code object.
        return cast(CodeType, code)

    def merge_after(self, op: Operation, other: "Lambda") -> "Lambda":
        """Combine this Lambda with another according to the given operation."""

        left = self.contextually_paren(op)
        right = other.contextually_paren(op, from_right=True)
        right = replace(right, tree=self.update_appended_closure_idxs(right.tree))
        return replace(
            left,
            args=left.args.merge(right.args),
            tree=op.render_ast(left.tree, right.tree),
            closure=self.closure.chain(right.closure),
            body_str=op.render_str(left, right),
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
        do_paren = self.last_op < op or (from_right and self.last_op <= op)
        if do_paren and self.last_op is not Operation.PARENS:
            return replace(
                self,
                body_str=Operation.PARENS.render_str(self),
                last_op=Operation.PARENS,
            )
        return self

    def apply_binary_op(
        self, op: Operation, other: Any, from_right: bool = False
    ) -> "Lambda":
        """Apply the given binary operation on this lambda"""
        new = self.contextually_paren(op, from_right)

        other_ast = subscript(name("closure"), constant(len(self.closure)))
        new_closure = self.closure.chain([other])

        if from_right:
            new_tree = op.render_ast(other_ast, new.tree)
            body_str = op.render_str(repr(other), new)
        else:
            new_tree = op.render_ast(new.tree, other_ast)
            body_str = op.render_str(new, repr(other))

        return replace(
            new,
            tree=new_tree,
            closure=new_closure,
            body_str=body_str,
            last_op=op,
        )

    def apply_unary_op(self, op: Operation) -> "Lambda":
        """Apply the given binary operation on this lambda"""
        new = self.contextually_paren(op, from_right=True)

        return replace(
            new,
            tree=op.render_ast(self.tree),
            body_str=op.render_str(new),
            last_op=op,
        )

    def apply_getattr(self, attr: str) -> "Lambda":
        """Apply the getattr operation on this lambda"""
        op = Operation.GETATTR
        new = self.contextually_paren(op)

        return replace(
            new,
            tree=op.render_ast(self.tree, attr),
            closure=self.closure,
            body_str=op.render_str(new, attr),
            last_op=op,
        )


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

    def __binary_op(
        self, op: Operation, other: Any, from_right: bool = False
    ) -> "LambdaBuilder":
        if isinstance(other, LambdaBuilder):
            # This situation means both sides are LambdaBuilder instances, and
            # since the left-side operations take precedence, from_right should
            # always be false here and doesn't need to be checked.

            # This swaps the original order of the operands (self = lhs, other
            # = rhs) to account for the asymmetry in the getitem operation:
            # we can only detect when this object is indexed, not when it's
            # used as an index. This is fine because the following block
            # detects a Lambda instance and always treats it as the rhs of any
            # operation.
            result = op(other, self.__lambda)
            # Operations on a LambdaBuilder (except call) always return another
            # LambdaBuilder instance, so this is safe.
            return cast(LambdaBuilder, result)

        if isinstance(other, Lambda):
            # Unless someone is mucking about with Lambda instances directly
            # rather than just letting this class handle them (they shouldn't)
            # this situation arises directly from the immediately preceding
            # code, but now we are in the right-side LambdaBuilder instance.
            return LambdaBuilder(
                other.merge_after(op, self.__lambda),
            )

        return LambdaBuilder(self.__lambda.apply_binary_op(op, other, from_right))

    def __bool__(self) -> NoReturn:
        raise NotImplementedError(
            "This object should not be directly tested for truthiness"
        )

    def __lt__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.LT, other)

    def __le__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.LE, other)

    def __eq__(self, other: Any) -> "LambdaBuilder":  # type: ignore
        return self.__binary_op(Operation.EQ, other)

    def __ne__(self, other: Any) -> "LambdaBuilder":  # type: ignore
        return self.__binary_op(Operation.NE, other)

    def __ge__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.GE, other)

    def __gt__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.GT, other)

    def __or__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.OR, other)

    def __ror__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.OR, other, from_right=True)

    def __xor__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.XOR, other)

    def __rxor__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.XOR, other, from_right=True)

    def __and__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.AND, other)

    def __rand__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.AND, other, from_right=True)

    def __lshift__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.LSHIFT, other)

    def __rlshift__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.LSHIFT, other, from_right=True)

    def __rshift__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.RSHIFT, other)

    def __rrshift__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.RSHIFT, other, from_right=True)

    def __add__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.ADD, other)

    def __radd__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.ADD, other, from_right=True)

    def __sub__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.SUB, other)

    def __rsub__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.SUB, other, from_right=True)

    def __mul__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.MUL, other)

    def __rmul__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.MUL, other, from_right=True)

    def __truediv__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.TRUEDIV, other)

    def __rtruediv__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.TRUEDIV, other, from_right=True)

    def __floordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.FLOORDIV, other)

    def __rfloordiv__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.FLOORDIV, other, from_right=True)

    def __mod__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.MOD, other)

    def __rmod__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.MOD, other, from_right=True)

    def __matmul__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.MATMUL, other)

    def __rmatmul__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.MATMUL, other, from_right=True)

    def __pow__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.POW, other)

    def __rpow__(self, other: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.POW, other, from_right=True)

    def __pos__(self) -> "LambdaBuilder":
        return LambdaBuilder(self.__lambda.apply_unary_op(Operation.POS))

    def __neg__(self) -> "LambdaBuilder":
        return LambdaBuilder(self.__lambda.apply_unary_op(Operation.NEG))

    def __invert__(self) -> "LambdaBuilder":
        return LambdaBuilder(self.__lambda.apply_unary_op(Operation.INVERT))

    def __getitem__(self, item: Any) -> "LambdaBuilder":
        return self.__binary_op(Operation.GETITEM, item)

    def __getattr__(self, item: str) -> "LambdaBuilder":
        return LambdaBuilder(self.__lambda.apply_getattr(item))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func = self.__compile()
        return func(*args, **kwargs)

    def __compile(self) -> Callable[..., Any]:
        code = self.__lambda.compile()
        closure = list(self.__lambda.closure)
        # Evaluating the AST we generate is key to the functioning of this
        # object, so we ignore the eval warning here.
        func = eval(code, {"closure": closure})  # pylint: disable=eval-used
        # This is guaranteed to be callable because the AST we are compiling
        # contains a single expression containing a single lambda accepting
        # one parameter.
        return cast(Callable[..., Any], func)


def uniq(iterable: Iterable[T]) -> Iterable[T]:
    """Imitate the Unix tool uniq, dedup adjacent items

    This function guarantees overall uniqueness of its elements if and only if
    the iterable argument is sorted first.

    """
    return map(itemgetter(0), groupby(iterable))
