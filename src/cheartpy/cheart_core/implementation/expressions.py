import dataclasses as dc
from typing import TextIO, Self

from cheartpy.cheart_core.interface.basis import _DataInterp
from ..interface import *


@dc.dataclass(slots=True)
class Expression(_Expression):
    name: str
    value: list[ExpressionValue | tuple[ExpressionValue, int]]

    def __repr__(self) -> str:
        return self.name

    def get_values(self):
        return self.value

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
