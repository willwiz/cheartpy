from .types import AbaqusEnum, CheartEnum, VtkEnum

Abaqus2Vtk = {
    AbaqusEnum.T3D2: VtkEnum.VtkLinearLine,
    AbaqusEnum.T3D3: VtkEnum.QuadLine,
    AbaqusEnum.CPS3: VtkEnum.LinTriangle,
    AbaqusEnum.CPS4: VtkEnum.LinQuadrilateral,
    AbaqusEnum.CPS4_3D: VtkEnum.QuadQuadrilateral,
    AbaqusEnum.C3D4: VtkEnum.LinHexahedron,
    AbaqusEnum.S3R: VtkEnum.LinTriangle,
    AbaqusEnum.TetQuad3D: VtkEnum.QuadTetrahedron,
    AbaqusEnum.Tet3D: VtkEnum.LinTetrahedron,
}

Abaqus2Cheart = {
    AbaqusEnum.T3D2: CheartEnum.LINE1,
    AbaqusEnum.T3D3: CheartEnum.LINE2,
    AbaqusEnum.CPS3: CheartEnum.LinTriangle,
    AbaqusEnum.CPS4: CheartEnum.LinQuadrilateral,
    AbaqusEnum.CPS4_3D: CheartEnum.QuadQuadrilateral,
    AbaqusEnum.C3D4: CheartEnum.LinHexahedron,
    AbaqusEnum.S3R: CheartEnum.LinTriangle,
    AbaqusEnum.TetQuad3D: CheartEnum.QuadTetrahedron,
    AbaqusEnum.Tet3D: CheartEnum.LinTetrahedron,
}

Abaqus2CheartNodeOrder = {
    AbaqusEnum.T3D2: [0, 1],
    AbaqusEnum.T3D3: [0, 1, 2],
    AbaqusEnum.CPS3: [0, 1, 2],
    AbaqusEnum.CPS4: [0, 1, 3, 2],
    AbaqusEnum.CPS4_3D: [0, 1, 3, 2, 4, 7, 8, 5, 6],
    AbaqusEnum.C3D4: [0, 1, 3, 2],
    AbaqusEnum.S3R: [0, 1, 2],
    AbaqusEnum.TetQuad3D: (0, 1, 3, 2, 4, 5, 7, 6),
    AbaqusEnum.Tet3D: [0, 1, 2, 3],
}
