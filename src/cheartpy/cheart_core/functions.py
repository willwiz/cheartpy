from .base_types import *


def create_cheart_basis(
    name: str, elem: CheartElementType | CHEART_ELEMENT_TYPE, kind: CheartBasisType | str,
    quadrature: CheartQuadratureType | str, order: int, gp: int,
) -> CheartBasis:
    elem = get_enum(elem, CheartElementType)
    kind = get_enum(kind, CheartBasisType)
    quadrature = get_enum(quadrature, CheartQuadratureType)
    if 2 * gp == order:
        raise ValueError(f"For {name}, order {order} <= {2 * gp - 1}")
    match quadrature, elem:
        case CheartQuadratureType.GAUSS_LEGENDRE, _:
            pass
        case CheartQuadratureType.KEAST_LYNESS, CheartElementType.TETRAHEDRAL_ELEMENT | CheartElementType.TRIANGLE_ELEMENT:
            pass
        case CheartQuadratureType.KEAST_LYNESS, _:
            raise ValueError(
                f"For {name} Basis, KEAST_LYNESS can only be used with tetrahydral or triangles")
    return CheartBasis(name, elem, Basis(kind, order), Quadrature(quadrature, gp))
