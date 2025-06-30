__all__ = ["Basis", "CheartBasis", "Quadrature"]
import dataclasses as dc
from typing import Literal, TextIO

from cheartpy.fe.aliases import CheartBasisType, CheartElementType, CheartQuadratureType
from cheartpy.fe.string_tools import join_fields
from cheartpy.fe.trait import IBasis, ICheartBasis, IQuadrature


@dc.dataclass(slots=True)
class Basis(IBasis):
    name: CheartBasisType
    _order: Literal[0, 1, 2]

    def __repr__(self) -> str:
        return f"{self.name}{self._order}"

    @property
    def order(self) -> Literal[0, 1, 2]:
        return self._order

    @property
    def kind(self) -> CheartBasisType:
        return self.name


@dc.dataclass(slots=True)
class Quadrature(IQuadrature):
    name: CheartQuadratureType
    _gp: int

    def __repr__(self) -> str:
        return self.name + str(self._gp)

    @property
    def gp(self) -> int:
        return self._gp

    @property
    def kind(self) -> CheartQuadratureType:
        return self.name


@dc.dataclass(slots=True)
class CheartBasis(ICheartBasis):
    name: str
    _elem: CheartElementType
    _basis: IBasis
    _quadrature: IQuadrature

    def __repr__(self) -> str:
        return self.name

    @property
    def elem(self) -> CheartElementType:
        return self._elem

    @property
    def basis(self) -> IBasis:
        return self._basis

    @property
    def quadrature(self) -> IQuadrature:
        return self._quadrature

    @property
    def order(self) -> Literal[0, 1, 2]:
        return self._basis.order

    @property
    def gp(self) -> int:
        return self._quadrature.gp

    def write(self, f: TextIO) -> None:
        string = join_fields(self.name, self._elem, self._basis, self._quadrature)
        f.write(f"!UseBasis={{{string}}}\n")
