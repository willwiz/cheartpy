import dataclasses as dc
from typing import TextIO

from ..pytools import join_fields
from ..trait.basis import *


@dc.dataclass(slots=True)
class DataPointer(IDataPointer):
    name: str
    file: str
    nt: int
    dim: int = 2

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        f.write(
            f"!DefDataPointer={{{join_fields(self.name,self.file,self.dim,self.nt)}}}\n"
        )


@dc.dataclass(slots=True)
class DataInterp(IDataInterp):
    var: IDataPointer

    def __repr__(self) -> str:
        return f"Interp({str(self.var)},t)"

    def get_datapointer(self) -> IDataPointer:
        return self.var
