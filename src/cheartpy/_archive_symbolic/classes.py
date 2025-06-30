from .types import Number, _Add_, _Div_, _Expression, _Fun_, _Mod_, _Mul_, _Pow_, _Sub_, _Symbol


class Symbol(_Symbol):
    def __add__(self, other: Number | _Symbol | _Expression) -> _Expression | _Symbol | Number:
        if isinstance(other, _Expression):
            return other.__add__(self)
        if isinstance(other, _Symbol):
            if other == self:
                return MultiplicationExpression(Number(2), self)
        return AdditionExpression(self, other)

    def __mul__(self, other: Number | _Symbol | _Expression) -> _Expression | _Symbol | Number:
        if isinstance(other, _Expression):
            return other.__mul__(self)
        if isinstance(other, Number):
            return MultiplicationExpression(other, self)
        if isinstance(other, _Symbol):
            if other == self:
                return PowerExpression(self, Number(2))
        return MultiplicationExpression(self, other)

    def __sub__(self, other: Number | _Symbol | _Expression) -> _Expression | _Symbol | Number:
        if isinstance(other, _Expression):
            return (other.__sub__(self)).__mul__(Number(-1))
        if isinstance(other, _Symbol):
            if other == self:
                return Number(0)
        return SubtractionExpression(self, other)

    def __div__(self, other: Number | _Symbol | _Expression) -> _Expression | _Symbol | Number:
        if isinstance(other, _Expression):
            return other.__sym_div__(self)
        if isinstance(other, Number):
            return DivisionExpression(self, other)
        if isinstance(other, _Symbol):
            if other == self:
                return Number(1)
        return DivisionExpression(self, other)

    def __pow__(self, other: Number | _Symbol | _Expression) -> _Expression | _Symbol | Number:
        if other == Number(1):
            return self
        return PowerExpression(self, other)

    def __mod__(self, other: Number | _Symbol | _Expression) -> _Expression | _Symbol | Number:
        if other == Number(1):
            return self
        return ModuloExpression(self, other)

    def mergeable(self, other) -> bool:
        if isinstance(other, _Symbol):
            return self == other
        if isinstance(other, _Expression):
            if other.op == _Add_() or other.op == _Sub_():
                return False
            if other.op == _Mul_():
                if other.term2 is None:
                    raise ValueError(f"{other} has invalid multiplier")
                return other.term1.mergeable(self) or other.term2.mergeable(self)
            if (
                other.op == _Div_()
                or other.op == _Pow_()
                or other.op == _Mod_()
                or isinstance(other.op, _Fun_)
            ):
                return False
        return False


class AdditionExpression(_Expression):
    def __init__(
        self,
        term1: _Expression | _Symbol | Number,
        term2: _Expression | _Symbol | Number,
    ) -> None:
        super().__init__(term1, _Add_(), term2)

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression:
        if isinstance(other, Number):
            if isinstance(self.term2, Number):
                return AdditionExpression(self.term1, self.term2 + other)
            if isinstance(self.term1, Number):
                return AdditionExpression(self.term2, self.term1 + other)
        if isinstance(other, _Symbol):
            if self.term1 == other:
                return AdditionExpression(self.term1.__mul__(other), self.term2)
            if self.term2 == other:
                return AdditionExpression(self.term1, self.term2.__mul__(other))
        if self == other:
            return MultiplicationExpression(Number(2), self)
        return AdditionExpression(self, other)

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __num_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __sym_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...


class SubtractionExpression(_Expression):
    def __init__(
        self,
        term1: _Expression | _Symbol | Number,
        term2: _Expression | _Symbol | Number,
    ) -> None:
        super().__init__(term1, _Sub_(), term2)

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __num_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __sym_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...


class MultiplicationExpression(_Expression):
    def __init__(
        self,
        term1: _Expression | _Symbol | Number,
        term2: _Expression | _Symbol | Number,
    ) -> None:
        super().__init__(term1, _Mul_(), term2)

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __num_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __sym_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...


class DivisionExpression(_Expression):
    def __init__(
        self,
        term1: _Expression | _Symbol | Number,
        term2: _Expression | _Symbol | Number,
    ) -> None:
        super().__init__(term1, _Div_(), term2)

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __num_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __sym_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...


class PowerExpression(_Expression):
    def __init__(
        self,
        term1: _Expression | _Symbol | Number,
        term2: _Expression | _Symbol | Number,
    ) -> None:
        super().__init__(term1, _Pow_(), term2)

    def __repr__(self) -> str:
        return f"{self.term1}**({self.term2})"

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __num_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __sym_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...


class ModuloExpression(_Expression):
    def __init__(
        self,
        term1: _Expression | _Symbol | Number,
        term2: _Expression | _Symbol | Number,
    ) -> None:
        super().__init__(term1, _Mod_(), term2)

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __num_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __sym_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...


class FunctionExpression(_Expression):
    def __init__(self, name: str, term1: _Expression | _Symbol | Number) -> None:
        super().__init__(term1, _Fun_(name), 1.0)

    def __repr__(self) -> str:
        return f"{self.op}({self.term1})"

    def __add__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __sub__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mul__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __div__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __num_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __sym_div__(
        self,
        other: Number | _Symbol | _Expression,
    ) -> Number | _Symbol | _Expression: ...

    def __pow__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...

    def __mod__(self, other: Number | _Symbol | _Expression) -> Number | _Symbol | _Expression: ...


def divide_symbol_expr(s: _Symbol, expr: _Expression) -> Number | _Symbol | _Expression:
    # match expr:
    #     case AdditionExpression():
    #         return DivisionExpression(s, expr)
    #     case SubtractionExpression():
    #         return DivisionExpression(s, expr)
    #     case MultiplicationExpression():
    #         if expr.term1.__div__(s) == Number(1):
    #             ...
    ...
