from typing import TYPE_CHECKING

from .types import AbaqusEnum, CheartEnum, VtkEnum

if TYPE_CHECKING:
    from collections.abc import Mapping

Abaqus2Vtk = {
    AbaqusEnum.T3D2: VtkEnum.VtkLinearLine,
    AbaqusEnum.T3D3: VtkEnum.VtkQuadraticLine,
    AbaqusEnum.S3R: VtkEnum.VtkLinearTriangle,
    AbaqusEnum.CPS3: VtkEnum.VtkLinearTriangle,
    AbaqusEnum.CPEG6: VtkEnum.VtkQuadraticTriangle,
    AbaqusEnum.CPS4: VtkEnum.VtkLinearQuadrilateral,
    AbaqusEnum.C3D4: VtkEnum.VtkLinearTetrahedron,
    AbaqusEnum.C3D10: VtkEnum.VtkQuadraticTetrahedron,
}

Abaqus2Cheart = {
    AbaqusEnum.T3D2: CheartEnum.LINE1,
    AbaqusEnum.T3D3: CheartEnum.LINE2,
    AbaqusEnum.S3R: CheartEnum.TRIANGLE1,
    AbaqusEnum.CPS3: CheartEnum.TRIANGLE1,
    AbaqusEnum.CPEG6: CheartEnum.TRIANGLE2,
    AbaqusEnum.CPS4: CheartEnum.QUADRILATERAL1,
    AbaqusEnum.C3D4: CheartEnum.TETRAHEDRON1,
    AbaqusEnum.C3D10: CheartEnum.TETRAHEDRON2,
}

Abaqus2CheartNodeOrder: Mapping[AbaqusEnum, tuple[int, ...]] = {
    AbaqusEnum.T3D2: (0, 1),
    AbaqusEnum.T3D3: (0, 1, 2),
    AbaqusEnum.S3R: (0, 1, 2),
    AbaqusEnum.CPS3: (0, 1, 2),
    AbaqusEnum.CPEG6: (0, 1, 2, 3, 5, 4),
    AbaqusEnum.CPS4: (0, 1, 2, 3),
    AbaqusEnum.C3D4: (0, 1, 2, 3),
    AbaqusEnum.C3D10: (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
}


def get_cheart_order_for_abaqus(elem: AbaqusEnum) -> tuple[int, ...]:
    return Abaqus2CheartNodeOrder[elem]
