# Star imports required to be at the module level
from onthelam.magic import *

def test_import_arbitrary_name():
    from onthelam.magic import foo, bar
    assert repr(foo + bar) == "bar, foo -> foo + bar"


def test_magic_underscore():
    import onthelam.magic as _
    assert repr(_.b[_.a]) == "a, b -> b[a]"


def test_auto_indexed_star_import():
    assert repr(_9[_8][_7][_6][_5][_4][_3][_2][_1][_0]) == (
        "_0, _1, _2, _3, _4, _5, _6, _7, _8, _9 -> _9[_8][_7][_6][_5][_4][_3][_2][_1][_0]"
    )
