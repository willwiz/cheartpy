__all__ = [
    "L2QMAP",
    "L2QMAPDICT",
    "_HEX_L2QMAP",
    "_LIN_L2QMAP",
    "_QUA_L2QMAP",
    "_TET_L2QMAP",
    "_TRI_L2QMAP",
]
from collections.abc import Collection, Sequence
from typing import Final

from cheartpy.vtk.struct import (
    VTKHEXAHEDRON2,
    VTKLINE2,
    VTKQUADRILATERAL2,
    VTKTETRAHEDRON2,
    VTKTRIANGLE2,
)
from cheartpy.vtk.types import VtkEnum

type L2QMAP = Sequence[Collection[int]]

_LIN_L2QMAP: Final[L2QMAP] = [
    {0},
    {1},
    {0, 1},
]
_TRI_L2QMAP: Final[L2QMAP] = [
    {0},
    {1},
    {2},
    {0, 1},
    {0, 2},
    {1, 2},
]
_QUA_L2QMAP: Final[L2QMAP] = [
    {0},
    {1},
    {2},
    {3},
    {0, 1},
    {0, 2},
    {0, 1, 2, 3},
    {1, 3},
    {2, 3},
]
_TET_L2QMAP: Final[L2QMAP] = [
    {0},
    {1},
    {2},
    {3},
    {0, 1},
    {0, 2},
    {1, 2},
    {0, 3},
    {1, 3},
    {2, 3},
]
_HEX_L2QMAP: Final[L2QMAP] = [
    {0},
    {1},
    {2},
    {3},
    {4},
    {5},
    {6},
    {7},
    {0, 1},
    {0, 2},
    {0, 1, 2, 3},
    {1, 3},
    {2, 3},
    {0, 4},
    {0, 1, 4, 5},
    {1, 5},
    {0, 2, 4, 6},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {1, 3, 5, 7},
    {2, 6},
    {2, 3, 6, 7},
    {3, 7},
    {4, 5},
    {4, 6},
    {4, 5, 6, 7},
    {5, 7},
    {6, 7},
]


L2QMAPDICT = {
    VtkEnum.LinLine: _LIN_L2QMAP,
    VtkEnum.LinTriangle: _TRI_L2QMAP,
    VtkEnum.LinQuadrilateral: _QUA_L2QMAP,
    VtkEnum.LinTetrahedron: _TET_L2QMAP,
    VtkEnum.LinHexahedron: _HEX_L2QMAP,
}
L2QTYPEDICT = {
    VtkEnum.LinLine: VTKLINE2,
    VtkEnum.LinTriangle: VTKTRIANGLE2,
    VtkEnum.LinQuadrilateral: VTKQUADRILATERAL2,
    VtkEnum.LinTetrahedron: VTKTETRAHEDRON2,
    VtkEnum.LinHexahedron: VTKHEXAHEDRON2,
}
