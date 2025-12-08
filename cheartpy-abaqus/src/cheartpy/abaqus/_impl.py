from cheartpy.vtk.trait import VtkType

from ._trait import AbaqusElement

_ABAQUS2VTK_TYPE_MAP = {
    AbaqusElement.T3D2: VtkType.LinLine,
    AbaqusElement.T3D3: VtkType.QuadLine,
    AbaqusElement.CPS3: VtkType.LinTriangle,
    AbaqusElement.CPS4: VtkType.LinQuadrilateral,
    AbaqusElement.CPS4_3D: VtkType.QuadQuadrilateral,
    AbaqusElement.C3D4: VtkType.LinHexahedron,
    AbaqusElement.S3R: VtkType.LinTriangle,
    AbaqusElement.TetQuad3D: VtkType.QuadTetrahedron,
    AbaqusElement.Tet3D: VtkType.LinTetrahedron,
}


def get_vtktype_from_abaqus_type(abaqus_elem: AbaqusElement) -> VtkType:
    # kind = _ABAQUS2VTK_TYPE_MAP.get(abaqus_elem)
    # if kind is None:
    #     msg = f"Unsupported Abaqus element type: {abaqus_elem}"
    return _ABAQUS2VTK_TYPE_MAP[abaqus_elem]
