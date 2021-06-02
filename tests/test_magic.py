# Star imports required to be at the module level
from onthelam.magic import *

def test_import_arbitrary_name():
    from onthelam.magic import foo, bar
    assert repr(foo + bar) == "bar, foo -> foo + bar"


def test_magic_underscore():
    import onthelam.magic as _
    assert repr(_.b[_.a]) == "a, b -> b[a]"
