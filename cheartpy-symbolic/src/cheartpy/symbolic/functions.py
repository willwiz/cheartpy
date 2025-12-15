from typing import Self

from .negatives import Negative
from .trait import ALL_TYPES, ExpressionTrait, FunctionTrait


class Function(FunctionTrait):
    __slots__ = ("_arg", "_name")
    _name: str
    _arg: list[ALL_TYPES]

    def __init__(self, name: str, *arg: ALL_TYPES) -> None:
        self._name = name
        self._arg = list(arg)

    def __str__(self) -> str:
        return f"{self._name}({self._arg!s})"

    def __hash__(self) -> int:
        return hash((self._name, tuple(self._arg)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Function):
            return False
        return self._name == other._name and self._arg == other._arg

    def __neg__(self) -> Negative[Self]:
        return Negative(self)

    def __add__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __radd__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __sub__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __rsub__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mul__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __rmul__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __div__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __rtruediv__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __pow__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __rpow__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mod__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __rmod__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> list[ALL_TYPES]:
        return self._arg


def sine(x: ALL_TYPES) -> Function:
    return Function("sin", x)


def cosine(x: ALL_TYPES) -> Function:
    return Function("cos", x)


def absolute(x: ALL_TYPES) -> Function:
    return Function("abs", x)


def exponent(x: ALL_TYPES) -> Function:
    return Function("exp", x)
