"""
The onthelam module provides a special object `_` which records operations
performed on it, returning an object that, when called, replays those
operations on the given argument. It also provides 10 additional numbered
instances (_0 through _9) to define lambdas accepting multiple arguments and
the LambdaBuilder class for constructing additional such objects.
"""

from .lambdabuilder import LambdaBuilder
from .magic import *

_ = LambdaBuilder("_")

# Pylint can't tell the dynamically generated variables are defined.
__all__ = [
    "_",
    "_0",  # pylint: disable=undefined-all-variable
    "_1",  # pylint: disable=undefined-all-variable
    "_2",  # pylint: disable=undefined-all-variable
    "_3",  # pylint: disable=undefined-all-variable
    "_4",  # pylint: disable=undefined-all-variable
    "_5",  # pylint: disable=undefined-all-variable
    "_6",  # pylint: disable=undefined-all-variable
    "_7",  # pylint: disable=undefined-all-variable
    "_8",  # pylint: disable=undefined-all-variable
    "_9",  # pylint: disable=undefined-all-variable
    "LambdaBuilder",
]
