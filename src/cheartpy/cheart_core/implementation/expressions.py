import dataclasses as dc
from typing import Sequence, TextIO, Self
from ..interface import *


@dc.dataclass(slots=True)
class Expression(_Expression):
    name: str
    value: Sequence[EXPRESSION_VALUE_TYPES]
    deps_var: dict[str, _Variable] = dc.field(default_factory=dict)
    deps_expr: dict[str, _Expression] = dc.field(default_factory=dict)

    def __repr__(self) -> str:
        return self.name

    def get_values(self):
        return self.value

    def add_expr_deps(self, *expr: _Expression):
        for v in expr:
            self.deps_expr[str(v)] = v

    def get_expr_deps(self):
        return self.deps_expr.values()

    def add_var_deps(self, *var: _Variable) -> None:
        for v in var:
            self.deps_var[str(v)] = v

    def get_var_deps(self):
        return self.deps_var.values()

    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def idx(self, key: int) -> str:
        return f"{self.name}.{key}"

    def write(self, f: TextIO):
        f.write(f"!DefExpression={{{self.name}}}\n")
        for v in self.value:
            if isinstance(v, tuple):
                f.write(f"  {v[0]!s}.{v[1]}\n")
            else:
                f.write(f"  {v!s}\n")
        f.write("\n")
