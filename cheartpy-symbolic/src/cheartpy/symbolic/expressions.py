from ._expression_algebra import add_expression
from .trait import ALL_TYPES, ExpressionTrait, ExpressionTuple, MathOperator


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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpressionTrait):
            return False
        return self._left == other.left and self._op == other.op and self._right == other.right

    def __neg__(self) -> ExpressionTrait:
        return Expression(-1, MathOperator.MUL, self)

    def __add__(self, other: ALL_TYPES) -> ALL_TYPES:
        vals = add_expression(self, other)
        return Expression(*vals)

    def __radd__(self, other: ALL_TYPES) -> ALL_TYPES:
        vals = add_expression(self, other)
        return Expression(vals.right, vals.op, vals.left)

    def __sub__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rsub__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __mul__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rmul__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __div__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rtruediv__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __pow__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rpow__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __mod__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    def __rmod__(self, other: ALL_TYPES) -> ALL_TYPES: ...
    @property
    def left(self) -> ALL_TYPES:
        return self._left

    @property
    def op(self) -> MathOperator:
        return self._op

    @property
    def right(self) -> ALL_TYPES:
        return self._right


def simplify_float_expression(left: float, op: MathOperator, right: float) -> float | int:
    match op:
        case MathOperator.ADD:
            return left + right
        case MathOperator.SUB:
            return left - right
        case MathOperator.MUL:
            return left * right
        case MathOperator.DIV:
            return left / right
        case MathOperator.POW:
            return left**right
        case MathOperator.MOD:
            return left % right


def convert_to_expression(val: ExpressionTuple) -> ALL_TYPES:
    if val.op is MathOperator.MUL:
        if val.left == 1:
            return val.right
        if val.right == 1:
            return val.left
    if isinstance(val.left, float | int) and isinstance(val.right, float | int):
        return simplify_float_expression(val.left, val.op, val.right)
    return Expression(val.left, val.op, val.right)
