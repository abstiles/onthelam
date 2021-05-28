import pytest

from onthelam.astoperations import Operation


def test_operation_ordering():
    assert Operation.ADD <= Operation.SUB < Operation.MUL
    assert Operation.PARENS > Operation.GT >= Operation.GE


def test_operation_ordering_same_class_only():
    with pytest.raises(TypeError):
        Operation.ADD < 20
    with pytest.raises(TypeError):
        Operation.ADD <= 10
    with pytest.raises(TypeError):
        Operation.ADD >= 10
    with pytest.raises(TypeError):
        Operation.ADD > 1

