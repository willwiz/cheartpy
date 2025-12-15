from typing import Self

from .negatives import Negative
from .trait import ALL_TYPES, ExpressionTrait, SymbolTrait, SymbolVal


class Symbol[T: SymbolVal](SymbolTrait):
    __slots__ = ("_value",)
    _value: T

    def __init__(self, value: T) -> None:
        self._value = value

    def __str__(self) -> str:
        return f"{self._value!s}"

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolTrait):
            return False
        return self._value == other.value

    def __neg__(self) -> Negative[Self]:
        return Negative(self)

    def __add__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __sub__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mul__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __div__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __pow__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mod__(self, other: ALL_TYPES) -> ExpressionTrait: ...

    @property
    def value(self) -> T:
        return self._value
