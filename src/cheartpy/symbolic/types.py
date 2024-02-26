from __future__ import annotations
import abc
from typing import Final


class _Operator_:
    op: Final[str]

    def __init__(self, op) -> None:
        self.op = op

    def __repr__(self) -> str:
        return self.op

    def __eq__(self, __value: _Operator_) -> bool:
        return self.op == __value.op


class _Add_(_Operator_):

    def __init__(self) -> None:
        super().__init__("+")


class _Sub_(_Operator_):

    def __init__(self) -> None:
        super().__init__("-")


class _Mul_(_Operator_):

    def __init__(self) -> None:
        super().__init__("*")


class _Div_(_Operator_):

    def __init__(self) -> None:
        super().__init__("/")


class _Pow_(_Operator_):

    def __init__(self) -> None:
        super().__init__("**")


class _Mod_(_Operator_):

    def __init__(self) -> None:
        super().__init__("%")


class _Fun_(_Operator_):

    def __init__(self, name: str) -> None:
        super().__init__(name)


class Number(abc.ABC):
    __slots__ = ["val"]
    val: float | int

    def __init__(self, val: float | int) -> None:
        self.val = val

    def __repr__(self) -> str:
        return str(self.val)

    def __eq__(self, other) -> bool:
        if isinstance(other, Number):
            return self.val == other.val
        return False

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression:
        if isinstance(other, Number):
            return type(self)(self.val + other.val)
        else:
            return other.__add__(self)

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression:
        if isinstance(other, Number):
            return type(self)(self.val - other.val)
        else:
            return other.__mul__(Number(-1)).__add__(self)

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression:
        if isinstance(other, Number):
            return type(self)(self.val * other.val)
        else:
            return other.__mul__(self)

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression:
        if isinstance(other, Number):
            return type(self)(self.val / other.val)
        elif isinstance(other, _Symbol):
            return _Expression(self, _Div_(), other)
        elif isinstance(other, _Expression):
            return other.__num_div__(self)
        raise ValueError("Number Division went wrong!!")

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression:
        if isinstance(other, Number):
            return type(self)(self.val ** other.val)
        else:
            return _Expression(self, _Pow_(), other)

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression:
        if isinstance(other, Number):
            return type(self)(self.val % other.val)
        else:
            return _Expression(self, _Mod_(), other)

    def mergeable(self, other) -> bool:
        if isinstance(other, Number):
            return True
        return False


class _Symbol:
    name: Final[str]

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        if isinstance(other, _Symbol):
            return self.name == other.name
        return False

    @abc.abstractmethod
    def __add__(self, other: Number | _Symbol |
                _Expression) -> _Expression: ...

    @abc.abstractmethod
    def __sub__(self, other: Number | _Symbol |
                _Expression) -> _Expression: ...

    @abc.abstractmethod
    def __mul__(self, other: Number | _Symbol |
                _Expression) -> _Expression: ...

    @abc.abstractmethod
    def __div__(self, other: Number | _Symbol |
                _Expression) -> _Expression: ...

    @abc.abstractmethod
    def __pow__(self, other: Number | _Symbol |
                _Expression) -> _Expression: ...

    @abc.abstractmethod
    def __mod__(self, other: Number | _Symbol |
                _Expression) -> _Expression: ...

    @abc.abstractmethod
    def mergeable(self, other) -> bool: ...


class _Expression:
    term1: _Expression | _Symbol | Number
    op: _Operator_
    term2: _Expression | _Symbol | Number

    def __init__(self, term1: _Expression | _Symbol | Number | int | float, op: _Operator_, term2: _Expression | _Symbol | Number | int | float) -> None:
        self.term1 = Number(term1) if isinstance(
            term1, (int, float)) else term1
        self.op = op
        self.term2 = Number(term2) if isinstance(
            term2, (int, float)) else term2

    def __repr__(self) -> str:
        return f"({str_expr(self.term1, self.op, self.term2)})"

    def __eq__(self, other) -> bool:
        if isinstance(other, _Expression):
            return self.term1 == other.term1 and self.op == other.op and self.term2 == other.term2
        return False

    @abc.abstractmethod
    def __add__(self, other: Number | _Symbol |
                _Expression) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def __sub__(self, other: Number | _Symbol |
                _Expression) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def __mul__(self, other: Number | _Symbol |
                _Expression) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def __div__(self, other: Number | _Symbol |
                _Expression) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def __num_div__(self, other: Number) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def __sym_div__(
        self, other: _Symbol) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def __pow__(self, other: Number | _Symbol |
                _Expression) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def __mod__(self, other: Number | _Symbol |
                _Expression) -> Number | _Symbol | _Expression: ...

    @abc.abstractmethod
    def mergeable(self, other) -> bool: ...


def str_expr(*vals: _Expression | _Symbol | _Operator_ | Number | None, char: str = " ") -> str:
    values = [str(v) for v in vals if v is not None]
    return char.join(values)
