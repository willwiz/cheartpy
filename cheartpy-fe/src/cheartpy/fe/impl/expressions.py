import dataclasses as dc
from typing import TYPE_CHECKING, Self, TextIO

from cheartpy.fe.trait import EXPRESSION_VALUE, IExpression, IVariable
from cheartpy.fe.utils.dependency_search import expression_get_expr_deps, expression_get_var_deps

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


@dc.dataclass(slots=True)
class Expression(IExpression):
    name: str
    value: Sequence[EXPRESSION_VALUE]
    var_deps: dict[str, IVariable] = dc.field(default_factory=dict[str, IVariable])
    expr_deps: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def values(self) -> Sequence[EXPRESSION_VALUE]:
        return self.value

    def add_deps(self, *var: IExpression | IVariable | None) -> None:
        for v in var:
            if isinstance(v, IExpression):
                self.add_expr_deps(v)
            elif isinstance(v, IVariable):
                self.add_var_deps(v)

    def add_expr_deps(self, *expr: IExpression) -> None:
        for v in expr:
            if str(v) not in self.expr_deps:
                self.expr_deps[str(v)] = v

    def add_var_deps(self, *var: IVariable) -> None:
        for v in var:
            if str(v) not in self.var_deps:
                self.var_deps[str(v)] = v

    def get_expr_deps(self) -> Collection[IExpression]:
        return expression_get_expr_deps(self)

    def get_var_deps(self) -> Collection[IVariable]:
        return expression_get_var_deps(self)

    def idx(self, key: int) -> str:
        return f"{self.name}.{key}"

    def write(self, f: TextIO) -> None:
        f.write(f"!DefExpression={{{self.name}}}\n")
        _values = (
            f"  {v[0]!s}.{v[1]}\n" if isinstance(v, tuple) else f"  {v!s}\n" for v in self.value
        )
        f.writelines(_values)
        f.write("\n")
