from ._types import GmshEnum
from ._vtk import VtkEnum

Vtk2Gmsh: dict[VtkEnum, GmshEnum] = {
    VtkEnum.LINE1: GmshEnum.LINE1,
    VtkEnum.TRIANGLE1: GmshEnum.TRIANGLE1,
    VtkEnum.QUADRILATERAL1: GmshEnum.QUADRILATERAL1,
    VtkEnum.TETRAHEDRON1: GmshEnum.TETRAHEDRON1,
    VtkEnum.HEXAHEDRON1: GmshEnum.HEXAHEDRON1,
    VtkEnum.LINE2: GmshEnum.LINE2,
    VtkEnum.TRIANGLE2: GmshEnum.TRIANGLE2,
    VtkEnum.QUADRILATERAL2: GmshEnum.QUADRILATERAL2,
    VtkEnum.TETRAHEDRON2: GmshEnum.TETRAHEDRON2,
    VtkEnum.HEXAHEDRON2: GmshEnum.HEXAHEDRON2,
}

Gmsh2Vtk: dict[GmshEnum, VtkEnum] = {v: k for k, v in Vtk2Gmsh.items()}


def get_gmsh_elem_from_vtk(vtk_elem: VtkEnum) -> GmshEnum | None:
    return Vtk2Gmsh.get(vtk_elem)


def get_vtk_elem_from_gmsh(gmsh_elem: GmshEnum) -> VtkEnum | None:
    return Gmsh2Vtk.get(gmsh_elem)
