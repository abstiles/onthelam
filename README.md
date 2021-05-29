# Onthelam

Tired of painstakingly writing the characters `lambda:` every time you want a
simple lambda function? **Onthelam** is the package for you.

```python
>>> from onthelam import _
>>> mapping = dict(foo=2, bar=1, baz=3)
>>> sorted_by_value = sorted(mapping.items(), key=_[1])
>>> sorted_by_value
[('bar', 1), ('foo', 2), ('baz', 3)]
```

Sure, you _could_ type `lambda pair: pair[1]`, but doesn't `_[1]` feel so much
nicer? I think it does.

Onthelam supports painless definition of lambda functions that use any
combination of comparators, arithmetic operations, bitwise operations,
indexing, and attribute getting.

## Installation and Requirements

Onthelam is installable from PyPI:

```shell
$ pip install on-the-lam
```

This initial release is only tested with and supports Python 3.9, but future
releases will aim at supporting older Python versions as well.

Onthelam is written in pure Python and brings in no additional dependencies.

## Readable repr

Onthelam lambdas provide a user-friendly `repr` string for easier debugging.

```python
>>> fn = -(_.count % 5 + 42) ** 3
>>> fn
_ -> -(_.count % 5 + 42) ** 3
```

Can your native lambdas do that?

This is especially useful for logging errors in functions that accept functions
as arguments, enabling you to log something about the argument that isn't just
`<function <lambda>(x)>`.

## Renamable

Maybe you take umbrage with my aesthetic choice to use an underscore as my
lambda identifier. That's fine. Rename it all you want:

```python
>>> from onthelam import LambdaBuilder
>>> λx = LambdaBuilder("λx")
>>> [*map(λx // 2, range(10)]
[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
```

It uses whatever name you give it in its `repr`:

```python
>>> λx // 2
λx -> λx // 2
```

## Composable

Need to use the same argument twice in your lambda? No sweat.

```pyrhon
>>> tetration_2 = _ ** _
>>> tetration_2(5)
3125
```

Need a lambda with more than one argument? Combining lambda builders with
different names gets you a function that takes as many arguments as you have
distinct names.

```python
>>> _a = LambdaBuilder("_a")
>>> _b = LambdaBuilder("_b")
>>> _b = LambdaBuilder("_c")
>>> fn = _a[_b] + _c
```

You still get a useful `repr`:

```python
>>> fn
_a, _b, _c -> _a[_b] + _c
```

And it works like you'd expect it:

```python
>>> fn([1, 2, 3], 2, 4)
7
```

Calling by keyword is permitted as well:

```python
>>> fn(_b=2, _c=4, _a=[1, 2, 3])
7
```

## Designed with itertools in mind

The itertools module in the standard library is incredibly powerful, but using
it often results in ugly code where you have to decide whether to use inline
lambdas which add a lot of line noise or lots of one-time named function
definition blocks that take up a lot of space relative to their importance.

Consider the following implementation of tetration, the mathematical operation
of iterated exponentiation of a number with itself.

```python
>>> from functools import reduce
>>> from itertools import repeat
>>> def tetration(x, n):
...     """Iterate `x ** x` n times"""
...     return reduce(_a ** _b, repeat(x, n))
...
>>> tetration(5, 1)
5
>>> tetration(5, 2)
3125
>>> tetration(5, 3)
298023223876953125
```

Onthelam clears out the clutter from using lambdas.

## Limitations

Onthelam works by using the various special methods available to a class for
customizing the behavior of an instance when it is operated on. In short:
through lots of operator overloading. The limitation is that there are some
expressions involving a `LambdaBuilder` instance that the object can't
seamlessly transform into a lambda function. As a result, they are interpreted
as attempts to use the defined lambda function as a lambda function. These are:

* Boolean contexts. Anything that tries to interpret the lambda argument's
  truthiness will fail entirely. E.g., the expression `1 if _ else 0` will fail.
* Use as the index to an object that is not itself a lambda argument. E.g., the
  expression `[1, 2, 3][_]` will fail.
* Tests of containment. E.g., the expression `_ in [1, 2, 3]` will fail.
* Using the lambda argument as an argument in a function call. E.g., the
  expression `ord(_)` will call the `ord` function with an identity lambda, not
  create a lambda which calls `ord` on its argument. Consider
  `functools.partial` for this case.
