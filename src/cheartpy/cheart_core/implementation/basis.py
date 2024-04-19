__all__ = ["Basis", "Quadrature", "CheartBasis"]
import dataclasses as dc
from typing import TextIO
from ..aliases import CheartBasisType, CheartElementType, CheartQuadratureType
from ..interface import *
from ..pytools import join_fields


@dc.dataclass(slots=True)
class Basis(_Basis):
    name: CheartBasisType
    order: int

    def __repr__(self):
        return self.name + str(self.order)


@dc.dataclass(slots=True)
class Quadrature(_Quadrature):
    name: CheartQuadratureType
    gp: int

    def __repr__(self):
        return self.name + str(self.gp)


@dc.dataclass(slots=True)
class CheartBasis(_CheartBasis):
    name: str
    elem: CheartElementType
    basis: Basis
    quadrature: Quadrature

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        string = join_fields(self.name, self.elem, self.basis, self.quadrature)
        f.write(f"!UseBasis={{{string}}}\n")
