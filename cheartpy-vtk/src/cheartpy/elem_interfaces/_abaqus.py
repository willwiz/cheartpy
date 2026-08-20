from typing import TYPE_CHECKING

from ._types import AbaqusEnum, CheartEnum, VtkEnum

if TYPE_CHECKING:
    from collections.abc import Mapping

Abaqus2Vtk = {
    AbaqusEnum.LINE1: VtkEnum.LINE1,
    AbaqusEnum.LINE2: VtkEnum.LINE2,
    AbaqusEnum.TRIANGLE1: VtkEnum.TRIANGLE1,
    AbaqusEnum.TRIANGLE2: VtkEnum.TRIANGLE2,
    AbaqusEnum.QUADRILATERAL1: VtkEnum.QUADRILATERAL1,
    AbaqusEnum.QUADRILATERAL2: VtkEnum.QUADRILATERAL2,
    AbaqusEnum.TETRAHEDRON1: VtkEnum.TETRAHEDRON1,
    AbaqusEnum.TETRAHEDRON2: VtkEnum.TETRAHEDRON2,
    AbaqusEnum.HEXAHEDRON1: VtkEnum.HEXAHEDRON1,
    AbaqusEnum.HEXAHEDRON2: VtkEnum.HEXAHEDRON2,
    AbaqusEnum.S3R: VtkEnum.TRIANGLE1,
    AbaqusEnum.CPEG6: VtkEnum.TRIANGLE2,
}
Vtk2Abaqus = {v: k for k, v in Abaqus2Vtk.items()}

_Abaqus2Cheart = {
    AbaqusEnum.LINE1: CheartEnum.LINE1,
    AbaqusEnum.LINE2: CheartEnum.LINE2,
    AbaqusEnum.S3R: CheartEnum.TRIANGLE1,
    AbaqusEnum.TRIANGLE1: CheartEnum.TRIANGLE1,
    AbaqusEnum.TRIANGLE2: CheartEnum.TRIANGLE2,
    AbaqusEnum.CPEG6: CheartEnum.TRIANGLE2,
    AbaqusEnum.QUADRILATERAL1: CheartEnum.QUADRILATERAL1,
    AbaqusEnum.TETRAHEDRON1: CheartEnum.TETRAHEDRON1,
    AbaqusEnum.TETRAHEDRON2: CheartEnum.TETRAHEDRON2,
}

_Abaqus2CheartNodeOrder: Mapping[AbaqusEnum, tuple[int, ...]] = {
    AbaqusEnum.LINE1: (0, 1),
    AbaqusEnum.LINE2: (0, 1, 2),
    AbaqusEnum.S3R: (0, 1, 2),
    AbaqusEnum.TRIANGLE1: (0, 1, 2),
    AbaqusEnum.TRIANGLE2: (0, 1, 2, 3, 5, 4),
    AbaqusEnum.CPEG6: (0, 1, 2, 3, 5, 4),
    AbaqusEnum.QUADRILATERAL1: (0, 1, 2, 3),
    AbaqusEnum.TETRAHEDRON1: (0, 1, 2, 3),
    AbaqusEnum.TETRAHEDRON2: (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
}

_AbaqusBoundaryElement: Mapping[AbaqusEnum, AbaqusEnum] = {
    AbaqusEnum.S3R: AbaqusEnum.LINE1,
    AbaqusEnum.TRIANGLE1: AbaqusEnum.LINE1,
    AbaqusEnum.TRIANGLE2: AbaqusEnum.LINE2,
    AbaqusEnum.CPEG6: AbaqusEnum.LINE2,
    AbaqusEnum.QUADRILATERAL1: AbaqusEnum.LINE1,
    AbaqusEnum.TETRAHEDRON1: AbaqusEnum.TRIANGLE1,
    AbaqusEnum.TETRAHEDRON2: AbaqusEnum.CPEG6,
}


def convert_abaqus_to_vtk(elem: AbaqusEnum) -> VtkEnum | None:
    return Abaqus2Vtk.get(elem)


def get_abaqus_elem_from_vtk(elem: VtkEnum) -> AbaqusEnum | None:
    return Vtk2Abaqus.get(elem)


def convert_abaqus_to_cheart(elem: AbaqusEnum) -> CheartEnum | None:
    return _Abaqus2Cheart.get(elem)


def get_vtk_element_for_abaqus(body: AbaqusEnum) -> VtkEnum | None:
    return Abaqus2Vtk.get(body)


def get_cheart_element_for_abaqus(body: AbaqusEnum) -> CheartEnum | None:
    return _Abaqus2Cheart.get(body)


def get_abaqus_boundary_element(body: AbaqusEnum) -> AbaqusEnum | None:
    return _AbaqusBoundaryElement.get(body)


def get_cheart_order_for_abaqus(elem: AbaqusEnum) -> tuple[int, ...]:
    return _Abaqus2CheartNodeOrder[elem]
