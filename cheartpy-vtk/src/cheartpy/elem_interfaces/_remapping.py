from typing import TYPE_CHECKING

import numpy as np

from ._abaqus import Abaqus2Vtk
from ._cheart import Vtk2Cheart
from ._gmsh import Vtk2Gmsh
from ._types import AbaqusEnum, VtkEnum

if TYPE_CHECKING:
    from pytools.arrays import A1

# fmt: off
Vtk2CheartNodeOrder: dict[VtkEnum, A1[np.intp]] = {
    VtkEnum.LINE1: np.array((0, 1)),
    VtkEnum.TRIANGLE1: np.array((0, 1, 2)),
    VtkEnum.QUADRILATERAL1: np.array((0, 1, 3, 2)),
    VtkEnum.TETRAHEDRON1: np.array((0, 1, 2, 3)),
    VtkEnum.HEXAHEDRON1: np.array((0, 1, 3, 2, 4, 5, 7, 6)),
    VtkEnum.LINE2: np.array((0, 1, 2)),
    VtkEnum.TRIANGLE2: np.array((0, 1, 2, 3, 5, 4)),
    VtkEnum.QUADRILATERAL2: np.array((0, 1, 3, 2, 4, 7, 8, 5, 6)),
    VtkEnum.TETRAHEDRON2: np.array((0, 1, 2, 3, 4, 6, 5, 7, 8, 9)),
    VtkEnum.HEXAHEDRON2: np.array((
         0,  1,  3,  2,  4,  5,  7,  6,
         8, 11, 24,  9, 10,
        16, 22, 17, 20, 26, 21, 19, 23, 18,
        12, 15, 25, 13, 14
    )),
}
# fmt: on
Cheart2VtkNodeOrder = {Vtk2Cheart[k]: np.argsort(v) for k, v in Vtk2CheartNodeOrder.items()}

Vtk2GmshNodeOrder: dict[VtkEnum, A1[np.intp]] = {
    VtkEnum.LINE1: np.arange(2),
    VtkEnum.TRIANGLE1: np.arange(3),
    VtkEnum.QUADRILATERAL1: np.arange(4),
    VtkEnum.TETRAHEDRON1: np.arange(4),
    VtkEnum.HEXAHEDRON1: np.arange(8),
    VtkEnum.LINE2: np.arange(3),
    VtkEnum.TRIANGLE2: np.arange(6),
    VtkEnum.QUADRILATERAL2: np.arange(9),
    VtkEnum.TETRAHEDRON2: np.arange(10),
    VtkEnum.HEXAHEDRON2: np.arange(20),
}
Gmsh2VtkNodeOrder = {Vtk2Gmsh[k]: v for k, v in Vtk2GmshNodeOrder.items()}

Abaqus2VtkNodeOrder: dict[AbaqusEnum, A1[np.intp]] = {
    AbaqusEnum.S3R: np.arange(3),
    AbaqusEnum.CPEG6: np.arange(6),
    AbaqusEnum.LINE1: np.arange(2),
    AbaqusEnum.LINE2: np.array([0, 2, 1]),
    AbaqusEnum.TRIANGLE1: np.arange(3),
    AbaqusEnum.TRIANGLE2: np.arange(6),
    AbaqusEnum.QUADRILATERAL1: np.arange(4),
    AbaqusEnum.QUADRILATERAL2: np.arange(9),
    AbaqusEnum.TETRAHEDRON1: np.arange(4),
    AbaqusEnum.TETRAHEDRON2: np.arange(10),
    AbaqusEnum.HEXAHEDRON1: np.arange(8),
    AbaqusEnum.HEXAHEDRON2: np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            26,
            24,
            23,
            25,
            21,
            22,
            20,
        ]
    ),
}
Vtk2AbaqusNodeOrder = {Abaqus2Vtk[k]: np.argsort(v) for k, v in Abaqus2VtkNodeOrder.items()}
