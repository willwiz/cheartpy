from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.fe.aliases import (
    CheartBasisEnum,
    CheartBasisType,
    CheartElementEnum,
    CheartElementType,
    CheartQuadratureEnum,
    CheartQuadratureType,
)
from cheartpy.fe.impl import Basis, CheartBasis, Quadrature
from cheartpy.fe.utils import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import ICheartBasis


_QUADRATURE_FOR_ELEM: dict[CheartElementEnum, CheartQuadratureType] = {
    CheartElementEnum.POINT_ELEMENT: "GAUSS_LEGENDRE",
    CheartElementEnum.point: "GAUSS_LEGENDRE",
    CheartElementEnum.ONED_ELEMENT: "GAUSS_LEGENDRE",
    CheartElementEnum.line: "GAUSS_LEGENDRE",
    CheartElementEnum.TRIANGLE_ELEMENT: "KEAST_LYNESS",
    CheartElementEnum.tri: "KEAST_LYNESS",
    CheartElementEnum.QUADRILATERAL_ELEMENT: "GAUSS_LEGENDRE",
    CheartElementEnum.quad: "GAUSS_LEGENDRE",
    CheartElementEnum.TETRAHEDRAL_ELEMENT: "KEAST_LYNESS",
    CheartElementEnum.tet: "KEAST_LYNESS",
    CheartElementEnum.HEXAHEDRAL_ELEMENT: "GAUSS_LEGENDRE",
    CheartElementEnum.hex: "GAUSS_LEGENDRE",
}


class _CreateBasisKwargs(TypedDict, total=False):
    quadrature: CheartQuadratureType
    gp: int


def create_basis(
    elem: CheartElementType,
    kind: CheartBasisType,
    order: int,
    **kwargs: Unpack[_CreateBasisKwargs],
) -> CheartBasis:
    _elem = get_enum(elem, CheartElementEnum)
    _kind = get_enum(kind, CheartBasisEnum)
    _quadrature = kwargs.get("quadrature", _QUADRATURE_FOR_ELEM[_elem])
    quadrature = get_enum(_quadrature, CheartQuadratureEnum)
    gp = kwargs.get("gp", 9 if quadrature is CheartQuadratureEnum.GAUSS_LEGENDRE else 4)
    if 2 * gp < order + 1:
        msg = f"For {_elem}, order {2 * gp} < {order + 1}"
        raise ValueError(msg)
    if quadrature is CheartQuadratureEnum.KEAST_LYNESS and elem not in [
        CheartElementEnum.TETRAHEDRAL_ELEMENT,
        CheartElementEnum.TRIANGLE_ELEMENT,
    ]:
        msg = f"For {_elem} Basis, KEAST_LYNESS can only be used with tetrahedral or triangles"
        raise ValueError(msg)
    return CheartBasis(_elem, Basis(_kind, order), Quadrature(quadrature, gp))


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
