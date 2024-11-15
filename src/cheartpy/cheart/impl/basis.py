__all__ = ["Basis", "Quadrature", "CheartBasis"]
import dataclasses as dc
from typing import Literal, TextIO
from ..aliases import CheartBasisType, CheartElementType, CheartQuadratureType
from ..trait import *
from ..pytools import join_fields


@dc.dataclass(slots=True)
class Basis(IBasis):
    name: CheartBasisType
    _order: Literal[1, 2]

    def __repr__(self):
        return self.name + str(self._order)

    @property
    def order(self) -> Literal[1, 2]:
        return self.order


@dc.dataclass(slots=True)
class Quadrature(IQuadrature):
    name: CheartQuadratureType
    gp: int

    def __repr__(self):
        return self.name + str(self.gp)


@dc.dataclass(slots=True)
class CheartBasis(ICheartBasis):
    name: str
    elem: CheartElementType
    basis: IBasis
    quadrature: IQuadrature

    def __repr__(self) -> str:
        return self.name

    @property
    def elem(self) -> CheartElementType:
        return self.elem

    @property
    def basis(self) -> IBasis:
        return self.basis

    @property
    def quadrature(self) -> IQuadrature:
        return self.quadrature

    @property
    def order(self) -> Literal[1, 2]:
        return self.basis.order

    def write(self, f: TextIO):
        string = join_fields(self.name, self.elem, self.basis, self.quadrature)
        f.write(f"!UseBasis={{{string}}}\n")
