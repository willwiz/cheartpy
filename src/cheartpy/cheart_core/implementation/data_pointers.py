import dataclasses as dc
from typing import TextIO
from ..interface.basis import *


@dc.dataclass(slots=True)
class DataPointer(_DataPointer):
    name: str
    file: str
    nt: int
    dim: int = 2

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        f.write(f"!DefDataPointer={{{self.name}|{
                self.file}|{self.dim}|{self.nt}}}\n")


@dc.dataclass(slots=True)
class DataInterp(_DataInterp):
    var: DataPointer

    def __repr__(self) -> str:
        return f"Interp({self.var.name},t)"
