import dataclasses as dc
from typing import TextIO


@dc.dataclass
class DataPointer:
    name: str
    file: str
    nt: int
    dim: int = 2

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        f.write(f"!DefDataPointer={{{self.name}|{
                self.file}|{self.dim}|{self.nt}}}\n")


@dc.dataclass
class DataInterp:
    var: DataPointer

    def __repr__(self) -> str:
        return f"Interp({self.var.name},t)"
