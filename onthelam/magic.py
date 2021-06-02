"""
Magically create new LambdaBuilder instances for any object imported from
this module.

Examples:

>>> from onthelam.magic import foo, bar
>>> foo + bar
"bar, foo -> foo + bar"

>>> import onthelam.magic as _
>>> _.b[_.a]
"a, b -> b[a]"
"""

import sys
from types import ModuleType
from typing import Any

from .lambdabuilder import LambdaBuilder


class DynamicModule(ModuleType):
    """Module type that generates a new LambdaBuilder on any module access"""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            return self.__getattribute__(name)
        return LambdaBuilder(name)

    @property
    def __all__(self):
        return []


sys.modules[__name__].__class__ = DynamicModule
