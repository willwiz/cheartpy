from typing import Self

from .negatives import Negative
from .trait import ALL_TYPES, ExpressionTrait, MathOperator


class Expression[L: ALL_TYPES, O: MathOperator, R: ALL_TYPES](ExpressionTrait):
    __slots__ = ("_left", "_op", "_right")
    _left: L
    _op: O
    _right: R

    def __init__(self, left: L, op: O, right: R) -> None:
        self._left = left
        self._op = op
        self._right = right

    def __str__(self) -> str:
        return f"({self._left!s} {self._op.value} {self._right!s})"

    def __hash__(self) -> int:
        return hash((self._left, self._op, self._right))

    def __eq__(self, other: object) -> bool: ...

    def __neg__(self) -> Negative[Self]:
        return Negative(self)

    def __add__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __sub__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mul__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __div__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __pow__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mod__(self, other: ALL_TYPES) -> ExpressionTrait: ...

    @property
    def left(self) -> ALL_TYPES:
        return self._left

    @property
    def op(self) -> MathOperator:
        return self._op

    @property
    def right(self) -> ALL_TYPES:
        return self._right
