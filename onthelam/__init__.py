"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument. It also provides the LambdaBuilder
class for constructing additional such objects.
"""

from .lambdabuilder import LambdaBuilder

_ = LambdaBuilder("_")

__all__ = ["_", "LambdaBuilder"]
