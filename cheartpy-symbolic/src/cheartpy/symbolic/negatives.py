from .trait import ALL_TYPES, ExpressionTrait, NegativeTrait


class Negative[T: ALL_TYPES](NegativeTrait):
    __slots__ = ("_value",)
    _value: T

    def __init__(self, value: T) -> None:
        self._value = value

    def __str__(self) -> str:
        return f"-{self._value!s}"

    def __hash__(self) -> int:
        return hash((-1, self._value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NegativeTrait):
            return False
        return self._value == other.value

    def __add__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __sub__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mul__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __div__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __pow__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __mod__(self, other: ALL_TYPES) -> ExpressionTrait: ...
    def __neg__(self) -> ALL_TYPES:
        return self._value

    @property
    def value(self) -> T:
        return self._value
