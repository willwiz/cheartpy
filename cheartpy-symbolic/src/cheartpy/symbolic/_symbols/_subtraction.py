# ruff: noqa: PLR0911
from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.scaled import Scaled
from cheartpy.symbolic.trait import (
    ALL_TYPES,
    ExpressionTrait,
    FunctionTrait,
    MathOperator,
    ScaledTrait,
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


def symbol_add_(left: SymbolTrait, right: ALL_TYPES) -> ALL_TYPES:
    match right:
        case float() | int():
            return Expression(left, MathOperator.ADD, right)
        case SymbolTrait():
            if left == right:
                return Scaled(2, left)
            return Expression(left, MathOperator.ADD, right)
        case FunctionTrait():
            return Expression(left, MathOperator.ADD, right)
        case ScaledTrait():
            if right.value == left:
                return Scaled(right.scalar + 1, left)
            return Expression(left, MathOperator.ADD, right)
        case ExpressionTrait():
            return symbol_add_expression_(left, right)


def symbol_add_expression_(left: SymbolTrait, right: ExpressionTrait) -> ALL_TYPES:
    match right.op:
        case MathOperator.ADD:
            return _symbol_add_expression_add_(left, right)
        case MathOperator.SUB:
            raise NotImplementedError
        case MathOperator.MUL:
            raise NotImplementedError
        case MathOperator.DIV:
            raise NotImplementedError
        case MathOperator.POW:
            raise NotImplementedError
        case MathOperator.MOD:
            raise NotImplementedError


def _symbol_add_expression_add_(left: SymbolTrait, right: ExpressionTrait) -> ALL_TYPES:
    # assert right.op is MathOperator.ADD
    if right.left == left:
        return Expression(Scaled(2, left), right.op, right.right)
    if right.right == left:
        return Expression(right.left, right.op, Scaled(2, left))
    if isinstance(right.left, ScaledTrait) and right.left.value == left:
        return Expression(Scaled(right.left.scalar + 1, left), right.op, right.right)
    if isinstance(right.right, ScaledTrait) and right.right.value == left:
        return Expression(right.left, right.op, Scaled(right.right.scalar + 1, left))
    return Expression(left, MathOperator.ADD, right)


def symbol_sub_(left: SymbolTrait, right: ALL_TYPES) -> ALL_TYPES:
    match right:
        case float() | int():
            return Expression(left, MathOperator.SUB, right)
        case SymbolTrait():
            if left == right:
                return 0
            return Expression(left, MathOperator.SUB, right)
        case FunctionTrait():
            return Expression(left, MathOperator.SUB, right)
        case ScaledTrait():
            if right.value == left:
                return -right + left
            return Expression(left, MathOperator.SUB, right)
        case ExpressionTrait():
            return Expression(left, MathOperator.SUB, right)
