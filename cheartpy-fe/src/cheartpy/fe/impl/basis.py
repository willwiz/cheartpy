import dataclasses as dc
from typing import Literal, TextIO

from cheartpy.fe.aliases import CheartBasisEnum, CheartElementEnum, CheartQuadratureEnum
from cheartpy.fe.trait import IBasis, ICheartBasis, IQuadrature

_ELEM_CODE = {
    CheartElementEnum.POINT_ELEMENT: "Pt",
    CheartElementEnum.point: "Pt",
    CheartElementEnum.ONED_ELEMENT: "Li",
    CheartElementEnum.line: "Li",
    CheartElementEnum.TRIANGLE_ELEMENT: "Tr",
    CheartElementEnum.tri: "Tr",
    CheartElementEnum.QUADRILATERAL_ELEMENT: "Qd",
    CheartElementEnum.quad: "Qd",
    CheartElementEnum.TETRAHEDRAL_ELEMENT: "Tt",
    CheartElementEnum.tet: "Tt",
    CheartElementEnum.HEXAHEDRAL_ELEMENT: "Hx",
    CheartElementEnum.hex: "Hx",
}

_BASIS_CODE = {
    CheartBasisEnum.NODAL_LAGRANGE: "Nl",
    CheartBasisEnum.NL: "Nl",
    CheartBasisEnum.MODAL_BASIS: "Mo",
    CheartBasisEnum.PNODAL_BASIS: "Pn",
    CheartBasisEnum.MINI_BASIS: "Mi",
    CheartBasisEnum.NURBS_BASIS: "Nu",
    CheartBasisEnum.SPECTRAL_BASIS: "Sp",
}

_ORDER_CODE = {0: "Z", 1: "L", 2: "Q", 3: "C", 4: "A", 5: "U"}

_QUAD_CODE = {
    CheartQuadratureEnum.GAUSS_LEGENDRE: "Gl",
    CheartQuadratureEnum.GL: "Gl",
    CheartQuadratureEnum.KEAST_LYNESS: "Kl",
    CheartQuadratureEnum.KL: "Kl",
}


@dc.dataclass(slots=True)
class Basis(IBasis):
    name: CheartBasisEnum
    _order: Literal[0, 1, 2]

    def __repr__(self) -> str:
        return f"{_ORDER_CODE[self._order]}{_BASIS_CODE[self.name]}"

    def __hash__(self) -> int:
        return hash((self.name, self._order))

    def __str__(self) -> str:
        return f"{self.name}{self._order}"

    @property
    def order(self) -> Literal[0, 1, 2]:
        return self._order

    @property
    def kind(self) -> CheartBasisEnum:
        return self.name


@dc.dataclass(slots=True)
class Quadrature(IQuadrature):
    name: CheartQuadratureEnum
    gp: int

    def __repr__(self) -> str:
        return f"{_QUAD_CODE[self.name]}{self.gp}"

    def __hash__(self) -> int:
        return hash((self.name, self.gp))

    def __str__(self) -> str:
        return f"{self.name}{self.gp}"

    @property
    def gp(self) -> int:
        return self.gp

    @property
    def kind(self) -> CheartQuadratureEnum:
        return self.name


def cheart_basis_name_code(elem: CheartElementEnum, quadrature: IQuadrature, basis: IBasis) -> str:
    return f"{_ELEM_CODE[elem]}{quadrature!r}{basis!r}"


@dc.dataclass(init=False, slots=True)
class CheartBasis(ICheartBasis):
    _elem: CheartElementEnum
    _basis: IBasis
    _quadrature: IQuadrature

    def __init__(self, elem: CheartElementEnum, basis: IBasis, quadrature: IQuadrature) -> None:
        self._elem = elem
        self._basis = basis
        self._quadrature = quadrature

    def __str__(self) -> str:
        return f"{_ELEM_CODE[self._elem]}{self._basis!r}{self._quadrature!r}"

    def __hash__(self) -> int:
        return hash((self._elem, self._basis, self._quadrature))

    def __repr__(self) -> str:
        return f"{self!s}|{self._elem!s}|{self._basis!s}|{self._quadrature!s}"

    @property
    def elem(self) -> CheartElementEnum:
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
        f.write(f"!UseBasis={{{self!r}}}\n")
