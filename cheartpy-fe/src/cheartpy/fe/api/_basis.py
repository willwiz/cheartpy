from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

from cheartpy.fe.aliases import (
    CheartBasisEnum,
    CheartBasisType,
    CheartElementEnum,
    CheartElementType,
    CheartQuadratureEnum,
)
from cheartpy.fe.impl import Basis, CheartBasis, Quadrature
from cheartpy.fe.utils import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import ICheartBasis

_ORDER = {0: "Z", 1: "L", 2: "Q", 3: "C", 4: "A", 5: "U"}

_ELEM = {
    CheartElementEnum.POINT_ELEMENT: "Point",
    CheartElementEnum.point: "Point",
    CheartElementEnum.ONED_ELEMENT: "Line",
    CheartElementEnum.line: "Line",
    CheartElementEnum.TRIANGLE_ELEMENT: "Tri",
    CheartElementEnum.tri: "Tri",
    CheartElementEnum.QUADRILATERAL_ELEMENT: "Quad",
    CheartElementEnum.quad: "Quad",
    CheartElementEnum.TETRAHEDRAL_ELEMENT: "Tet",
    CheartElementEnum.tet: "Tet",
    CheartElementEnum.HEXAHEDRAL_ELEMENT: "Hex",
    CheartElementEnum.hex: "Hex",
}

_QUADRATURE_FOR_ELEM: dict[CheartElementEnum, CheartQuadratureEnum] = {
    CheartElementEnum.POINT_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.point: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.ONED_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.line: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.TRIANGLE_ELEMENT: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.tri: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.QUADRILATERAL_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.quad: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.TETRAHEDRAL_ELEMENT: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.tet: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.HEXAHEDRAL_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.hex: CheartQuadratureEnum.GAUSS_LEGENDRE,
}


class _CreateBasisKwargs(TypedDict, total=False):
    gp: int


def create_basis(
    elem: CheartElementType,
    kind: CheartBasisType,
    order: Literal[0, 1, 2],
    **kwargs: Unpack[_CreateBasisKwargs],
) -> CheartBasis:
    _elem = get_enum(elem, CheartElementEnum)
    _kind = get_enum(kind, CheartBasisEnum)
    quadrature = _QUADRATURE_FOR_ELEM[_elem]
    name = f"{_ORDER[order]}{_ELEM[_elem]}"
    gp = kwargs.get("gp", 9 if quadrature is CheartQuadratureEnum.GAUSS_LEGENDRE else 4)
    if 2 * gp < order + 1:
        msg = f"For {name}, order {2 * gp} < {order + 1}"
        raise ValueError(msg)
    if quadrature is CheartQuadratureEnum.KEAST_LYNESS and elem not in [
        CheartElementEnum.TETRAHEDRAL_ELEMENT,
        CheartElementEnum.TRIANGLE_ELEMENT,
    ]:
        msg = f"For {name} Basis, KEAST_LYNESS can only be used with tetrahedral or triangles"
        raise ValueError(msg)
    return CheartBasis(name, _elem, Basis(_kind, order), Quadrature(quadrature, gp))


def create_boundary_basis(vol: ICheartBasis) -> CheartBasis:
    match vol.elem:
        case CheartElementEnum.HEXAHEDRAL_ELEMENT | CheartElementEnum.hex:
            elem = "QUADRILATERAL_ELEMENT"
        case CheartElementEnum.TETRAHEDRAL_ELEMENT | CheartElementEnum.tet:
            elem = "TRIANGLE_ELEMENT"
        case CheartElementEnum.QUADRILATERAL_ELEMENT | CheartElementEnum.quad:
            elem = "ONED_ELEMENT"
        case CheartElementEnum.TRIANGLE_ELEMENT | CheartElementEnum.tri:
            elem = "ONED_ELEMENT"
        case CheartElementEnum.ONED_ELEMENT | CheartElementEnum.line:
            elem = "POINT_ELEMENT"
        case CheartElementEnum.POINT_ELEMENT | CheartElementEnum.point:
            msg = "No such thing as boundary for point elements"
            raise ValueError(msg)
    return create_basis(elem, vol.basis.kind.value, vol.basis.order, gp=vol.quadrature.gp)
