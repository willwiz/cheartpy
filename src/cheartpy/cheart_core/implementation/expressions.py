import dataclasses as dc
from typing import TextIO, Self

from .data_pointers import DataInterp


@dc.dataclass
class Expression:
    name: str
    value: list[str | float | DataInterp]

    def __repr__(self) -> str:
        return self.name

    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def idx(self, key: int) -> str:
        return f"{self.name}.{key}"

    def write(self, f: TextIO):
        f.write(f"!DefExpression={{{self.name}}}\n")
        for v in self.value:
            f.write(f"  {v!s}\n")
        f.write("\n")
