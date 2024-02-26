import dataclasses as dc
from typing import TextIO
from .aliases import CheartBasisType, CheartElementType, CheartQuadratureType
from .pytools import join_fields


@dc.dataclass(slots=True)
class Basis:
    name: CheartBasisType
    order: int

    def to_str(self):
        return self.name+str(self.order)


@dc.dataclass(slots=True)
class Quadrature:
    name: CheartQuadratureType
    gp: int

    def to_str(self):
        return self.name+str(self.gp)


@dc.dataclass
class CheartBasis:
    name: str
    elem: CheartElementType
    basis: Basis
    quadrature: Quadrature

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        string = join_fields(self.name, self.elem, self.basis.to_str(
        ), self.quadrature.to_str())
        f.write(
            f"!UseBasis={{{string}}}\n"
        )
