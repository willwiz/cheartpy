from ._types import CheartEnum, VtkEnum

Vtk2Cheart = {
    VtkEnum.LINE1: CheartEnum.LINE1,
    VtkEnum.TRIANGLE1: CheartEnum.TRIANGLE1,
    VtkEnum.QUADRILATERAL1: CheartEnum.QUADRILATERAL1,
    VtkEnum.TETRAHEDRON1: CheartEnum.TETRAHEDRON1,
    VtkEnum.HEXAHEDRON1: CheartEnum.HEXAHEDRON1,
    VtkEnum.LINE2: CheartEnum.LINE2,
    VtkEnum.TRIANGLE2: CheartEnum.TRIANGLE2,
    VtkEnum.QUADRILATERAL2: CheartEnum.QUADRILATERAL2,
    VtkEnum.TETRAHEDRON2: CheartEnum.TETRAHEDRON2,
    VtkEnum.HEXAHEDRON2: CheartEnum.HEXAHEDRON2,
}
Cheart2Vtk = {v: k for k, v in Vtk2Cheart.items()}


def get_cheart_elem_from_vtk(elem: VtkEnum) -> CheartEnum | None:
    return Vtk2Cheart.get(elem)
