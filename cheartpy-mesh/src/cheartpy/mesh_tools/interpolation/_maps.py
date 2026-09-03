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

from cheartpy.elem_interfaces import CheartEnum
from cheartpy.vtk import get_vtk_elem

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
    CheartEnum.LINE1: _LIN_L2QMAP,
    CheartEnum.TRIANGLE1: _TRI_L2QMAP,
    CheartEnum.QUADRILATERAL1: _QUA_L2QMAP,
    CheartEnum.TETRAHEDRON1: _TET_L2QMAP,
    CheartEnum.HEXAHEDRON1: _HEX_L2QMAP,
}
L2QTYPEDICT = {
    CheartEnum.LINE1: get_vtk_elem(CheartEnum.LINE2),
    CheartEnum.TRIANGLE1: get_vtk_elem(CheartEnum.TRIANGLE2),
    CheartEnum.QUADRILATERAL1: get_vtk_elem(CheartEnum.QUADRILATERAL2),
    CheartEnum.TETRAHEDRON1: get_vtk_elem(CheartEnum.TETRAHEDRON2),
    CheartEnum.HEXAHEDRON1: get_vtk_elem(CheartEnum.HEXAHEDRON2),
}
