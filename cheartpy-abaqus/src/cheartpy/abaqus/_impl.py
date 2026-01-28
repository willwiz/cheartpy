from cheartpy.vtk.types import VtkEnum

from ._trait import AbaqusElement

_ABAQUS2VTK_TYPE_MAP = {
    AbaqusElement.T3D2: VtkEnum.LinLine,
    AbaqusElement.T3D3: VtkEnum.QuadLine,
    AbaqusElement.CPS3: VtkEnum.LinTriangle,
    AbaqusElement.CPS4: VtkEnum.LinQuadrilateral,
    AbaqusElement.CPS4_3D: VtkEnum.QuadQuadrilateral,
    AbaqusElement.C3D4: VtkEnum.LinHexahedron,
    AbaqusElement.S3R: VtkEnum.LinTriangle,
    AbaqusElement.TetQuad3D: VtkEnum.QuadTetrahedron,
    AbaqusElement.Tet3D: VtkEnum.LinTetrahedron,
}


def get_vtktype_from_abaqus_enum(abaqus_elem: AbaqusElement) -> VtkEnum:
    # kind = _ABAQUS2VTK_TYPE_MAP.get(abaqus_elem)
    # if kind is None:
    #     msg = f"Unsupported Abaqus element type: {abaqus_elem}"
    return _ABAQUS2VTK_TYPE_MAP[abaqus_elem]
