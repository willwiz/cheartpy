from typing import TYPE_CHECKING

from ._types import AbaqusEnum, CheartEnum, GmshEnum, VtkEnum

if TYPE_CHECKING:
    from collections.abc import Mapping

type SumType = CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
CHEART_2_VTK: Mapping[SumType, SumType] = {
    CheartEnum.LINE1: VtkEnum.LINE1,
    CheartEnum.TRIANGLE1: VtkEnum.TRIANGLE1,
    CheartEnum.QUADRILATERAL1: VtkEnum.QUADRILATERAL1,
    CheartEnum.TETRAHEDRON1: VtkEnum.TETRAHEDRON1,
    CheartEnum.HEXAHEDRON1: VtkEnum.HEXAHEDRON1,
    CheartEnum.LINE2: VtkEnum.LINE2,
    CheartEnum.TRIANGLE2: VtkEnum.TRIANGLE2,
    CheartEnum.QUADRILATERAL2: VtkEnum.QUADRILATERAL2,
    CheartEnum.TETRAHEDRON2: VtkEnum.TETRAHEDRON2,
    CheartEnum.HEXAHEDRON2: VtkEnum.HEXAHEDRON2,
}
VTK_2_CHEART: Mapping[SumType, SumType] = {v: k for k, v in CHEART_2_VTK.items()}

ABAQUS_2_VTK: Mapping[SumType, SumType] = {
    AbaqusEnum.S3R: VtkEnum.TRIANGLE1,
    AbaqusEnum.CPEG6: VtkEnum.TRIANGLE2,
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
}
VTK_2_ABAQUS: Mapping[SumType, SumType] = {v: k for k, v in ABAQUS_2_VTK.items()}
GMSH_2_VTK: Mapping[SumType, SumType] = {
    GmshEnum.LINE1: VtkEnum.LINE1,
    GmshEnum.LINE2: VtkEnum.LINE2,
    GmshEnum.TRIANGLE1: VtkEnum.TRIANGLE1,
    GmshEnum.TRIANGLE2: VtkEnum.TRIANGLE2,
    GmshEnum.QUADRILATERAL1: VtkEnum.QUADRILATERAL1,
    GmshEnum.QUADRILATERAL2: VtkEnum.QUADRILATERAL2,
    GmshEnum.TETRAHEDRON1: VtkEnum.TETRAHEDRON1,
    GmshEnum.TETRAHEDRON2: VtkEnum.TETRAHEDRON2,
    GmshEnum.HEXAHEDRON1: VtkEnum.HEXAHEDRON1,
    GmshEnum.HEXAHEDRON2: VtkEnum.HEXAHEDRON2,
}
VTK_2_GMSH: Mapping[SumType, SumType] = {v: k for k, v in GMSH_2_VTK.items()}

ABAQUS_2_CHEART: Mapping[SumType, SumType] = {k: VTK_2_CHEART[v] for k, v in ABAQUS_2_VTK.items()}
CHEART_2_ABAQUS: Mapping[SumType, SumType] = {v: k for k, v in ABAQUS_2_CHEART.items()}
GMSH_2_CHEART: Mapping[SumType, SumType] = {k: VTK_2_CHEART[v] for k, v in GMSH_2_VTK.items()}
CHEART_2_GMSH: Mapping[SumType, SumType] = {v: k for k, v in GMSH_2_CHEART.items()}
ABAQUS_2_GMSH: Mapping[SumType, SumType] = {k: VTK_2_GMSH[v] for k, v in ABAQUS_2_VTK.items()}
GMSH_2_ABAQUS: Mapping[SumType, SumType] = {v: k for k, v in ABAQUS_2_GMSH.items()}


TYPE_MAP: Mapping[type[SumType], Mapping[type[SumType], Mapping[SumType, SumType]]] = {
    CheartEnum: {
        VtkEnum: CHEART_2_VTK,
        AbaqusEnum: CHEART_2_ABAQUS,
        GmshEnum: CHEART_2_GMSH,
    },
    VtkEnum: {
        CheartEnum: VTK_2_CHEART,
        AbaqusEnum: VTK_2_ABAQUS,
        GmshEnum: VTK_2_GMSH,
    },
    AbaqusEnum: {
        CheartEnum: ABAQUS_2_CHEART,
        VtkEnum: ABAQUS_2_VTK,
        GmshEnum: ABAQUS_2_GMSH,
    },
    GmshEnum: {
        CheartEnum: GMSH_2_CHEART,
        VtkEnum: GMSH_2_VTK,
        AbaqusEnum: GMSH_2_ABAQUS,
    },
}


def convert_element_type(input_type: SumType, target: type[SumType]) -> SumType:
    return TYPE_MAP[type(input_type)][target][input_type]
