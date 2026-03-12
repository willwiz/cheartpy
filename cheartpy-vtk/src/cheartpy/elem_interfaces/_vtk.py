from pytools.result import Err, Ok, Result

from ._types import CheartEnum, VtkEnum

Vtk2Cheart = {
    VtkEnum.VtkLinearLine: CheartEnum.LINE1,
    VtkEnum.VtkLinearTriangle: CheartEnum.TRIANGLE1,
    VtkEnum.VtkLinearQuadrilateral: CheartEnum.QUADRILATERAL1,
    VtkEnum.VtkLinearTetrahedron: CheartEnum.TETRAHEDRON1,
    VtkEnum.VtkLinearHexahedron: CheartEnum.HEXAHEDRON1,
    VtkEnum.VtkQuadraticLine: CheartEnum.LINE2,
    VtkEnum.VtkQuadraticTriangle: CheartEnum.TRIANGLE2,
    VtkEnum.VtkQuadraticQuadrilateral: CheartEnum.QUADRILATERAL2,
    VtkEnum.VtkQuadraticTetrahedron: CheartEnum.TETRAHEDRON2,
    VtkEnum.VtkQuadraticHexahedron: CheartEnum.HEXAHEDRON2,
}


# fmt: off
Vtk2CheartNodeOrder = {
    VtkEnum.VtkLinearLine: (0, 1),
    VtkEnum.VtkLinearTriangle: (0, 1, 2),
    VtkEnum.VtkLinearQuadrilateral: (0, 1, 3, 2),
    VtkEnum.VtkLinearTetrahedron: (0, 1, 2, 3),
    VtkEnum.VtkLinearHexahedron: (0, 1, 5, 4, 2, 3, 7, 6),
    VtkEnum.VtkQuadraticLine: (0, 1, 2),
    VtkEnum.VtkQuadraticTriangle: (0, 1, 2, 3, 5, 4),
    VtkEnum.VtkQuadraticQuadrilateral: (0, 1, 3, 2, 4, 7, 8, 5, 6),
    VtkEnum.VtkQuadraticTetrahedron: (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
    VtkEnum.VtkQuadraticHexahedron: (
        0,  1,  5,  4,  2,  3,  7, 6,  8,  15,
        22, 13, 12, 21, 26, 19, 9, 11, 25, 23,
        16, 18, 10, 24, 14, 20, 17,
    ),
}
# fmt: on

VtkBoundaryElement: dict[VtkEnum, VtkEnum] = {
    VtkEnum.VtkLinearTriangle: VtkEnum.VtkLinearLine,
    VtkEnum.VtkLinearQuadrilateral: VtkEnum.VtkLinearLine,
    VtkEnum.VtkLinearTetrahedron: VtkEnum.VtkLinearTriangle,
    VtkEnum.VtkLinearHexahedron: VtkEnum.VtkLinearQuadrilateral,
    VtkEnum.VtkQuadraticTriangle: VtkEnum.VtkQuadraticLine,
    VtkEnum.VtkQuadraticQuadrilateral: VtkEnum.VtkQuadraticLine,
    VtkEnum.VtkQuadraticTetrahedron: VtkEnum.VtkQuadraticTriangle,
    VtkEnum.VtkQuadraticHexahedron: VtkEnum.VtkQuadraticQuadrilateral,
}


def get_cheart_order_for_vtk(elem: VtkEnum) -> tuple[int, ...]:
    return Vtk2CheartNodeOrder[elem]


def get_vtk_boundary_element(elem: VtkEnum) -> VtkEnum | None:
    return VtkBoundaryElement.get(elem)


def guess_vtk_elem_from_dim(edim: int, bdim: int | None) -> Result[VtkEnum]:
    match edim, bdim:
        case 3, 2 | None:
            elem = VtkEnum.VtkLinearTriangle
        case 6, 3 | None:
            elem = VtkEnum.VtkQuadraticTriangle
        case 4, 2:
            elem = VtkEnum.VtkLinearQuadrilateral
        case 9, 3 | None:
            elem = VtkEnum.VtkQuadraticQuadrilateral
        case 4, 3:
            elem = VtkEnum.VtkLinearTetrahedron
        case 10, 6 | None:
            elem = VtkEnum.VtkQuadraticTetrahedron
        case 8, 4 | None:
            elem = VtkEnum.VtkLinearHexahedron
        case 27, 9 | None:
            elem = VtkEnum.VtkQuadraticHexahedron
        case 4, None:
            msg = (
                "Cannot detect between Bilinear quadrilateral and Trilinear tetrahedron,"
                "need boundary dim"
            )
            return Err(ValueError(msg))
        case _:
            msg = f"Unsupported element dimensions: edim={edim}, bdim={bdim}"
            return Err(ValueError(msg))
    return Ok(elem)
