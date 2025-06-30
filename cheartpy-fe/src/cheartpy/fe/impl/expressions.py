__all__ = ["Expression"]
import dataclasses as dc
from collections.abc import Sequence, ValuesView
from typing import Self, TextIO

from cheartpy.fe.trait import EXPRESSION_VALUE, IExpression, IVariable


@dc.dataclass(slots=True)
class Expression(IExpression):
    name: str
    value: Sequence[EXPRESSION_VALUE]
    deps_var: dict[str, IVariable] = dc.field(default_factory=dict[str, IVariable])
    deps_expr: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])

    def __repr__(self) -> str:
        return self.name

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def get_values(self) -> Sequence[EXPRESSION_VALUE]:
        return self.value

    def add_deps(self, *var: IExpression | IVariable | None) -> None:
        for v in var:
            if isinstance(v, IExpression):
                self.add_expr_deps(v)
            elif isinstance(v, IVariable):
                self.add_var_deps(v)

    def add_expr_deps(self, *expr: IExpression) -> None:
        for v in expr:
            if str(v) not in self.deps_expr:
                self.deps_expr[str(v)] = v

    def get_expr_deps(self) -> ValuesView[IExpression]:
        return self.deps_expr.values()

    def add_var_deps(self, *var: IVariable) -> None:
        for v in var:
            if str(v) not in self.deps_var:
                self.deps_var[str(v)] = v

    def get_var_deps(self) -> ValuesView[IVariable]:
        return self.deps_var.values()

    def idx(self, key: int) -> str:
        return f"{self.name}.{key}"

    def write(self, f: TextIO) -> None:
        f.write(f"!DefExpression={{{self.name}}}\n")
        for v in self.value:
            if isinstance(v, tuple):
                f.write(f"  {v[0]!s}.{v[1]}\n")
            else:
                f.write(f"  {v!s}\n")
        f.write("\n")
