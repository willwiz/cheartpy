from ._symbols import symbol_add_, symbol_sub_
from .expressions import Expression
from .trait import (
    ALL_TYPES,
    ExpressionTrait,
    MathOperator,
    SymbolTrait,
    SymbolVal,
)


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

    def __neg__(self) -> ExpressionTrait:
        return Expression(-1, MathOperator.MUL, self)

    def __add__(self, other: ALL_TYPES) -> ALL_TYPES:
        return symbol_add_(self, other)

    def __radd__(self, other: ALL_TYPES) -> ALL_TYPES:
        if isinstance(other, (float, int)):
            return Expression(other, MathOperator.ADD, self)
        return other + self

    def __sub__(self, other: ALL_TYPES) -> ALL_TYPES:
        return symbol_sub_(self, other)

    def __rsub__(self, other: ALL_TYPES) -> ALL_TYPES:
        if isinstance(other, (float, int)):
            return Expression(other, MathOperator.SUB, self)
        return other - self

    def __mul__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rmul__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __div__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rtruediv__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __pow__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rpow__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __mod__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rmod__(self, other: ALL_TYPES) -> ALL_TYPES: ...

    @property
    def value(self) -> T:
        return self._value
