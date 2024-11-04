__all__ = [
    "_LIN_L2QMAP",
    "_TRI_L2QMAP",
    "_QUA_L2QMAP",
    "_TET_L2QMAP",
    "_HEX_L2QMAP",
    "L2QMAP",
    "L2QMAPDICT",
]
from typing import Collection, Final, Sequence

from ...cheart_mesh.elements import VtkType

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
    VtkType.LineLinear: _LIN_L2QMAP,
    VtkType.TriangleLinear: _TRI_L2QMAP,
    VtkType.QuadrilateralLinear: _QUA_L2QMAP,
    VtkType.TetrahedronLinear: _TET_L2QMAP,
    VtkType.HexahedronLinear: _HEX_L2QMAP,
}
L2QTYPEDICT = {
    VtkType.LineLinear: VtkType.LineQuadratic,
    VtkType.TriangleLinear: VtkType.TriangleQuadratic,
    VtkType.QuadrilateralLinear: VtkType.QuadrilateralQuadratic,
    VtkType.TetrahedronLinear: VtkType.TetrahedronQuadratic,
    VtkType.HexahedronLinear: VtkType.HexahedronQuadratic,
}
