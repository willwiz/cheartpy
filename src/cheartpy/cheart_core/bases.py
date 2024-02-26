import dataclasses as dc
from typing import TextIO
from .aliases import CheartBasisType, CheartElementType, CheartQuadratureType
from .pytools import join_fields


@dc.dataclass(slots=True)
class Basis:
    name: CheartBasisType
    order: int


@dc.dataclass(slots=True)
class Quadrature:
    name: CheartQuadratureType
    gp: int


@dc.dataclass
class CheartBasis:
    name: str
    elem: CheartElementType
    basis: Basis
    quadrature: Quadrature

    def write(self, f: TextIO):
        string = join_fields([self.name, self.elem, self.basis.name,
                             self.basis.order, self.quadrature.name, self.quadrature.gp])
        f.write(
            f"!UseBasis={{{string}}}\n"
        )
