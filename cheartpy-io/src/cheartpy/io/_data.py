import abc
from typing import TYPE_CHECKING, TextIO

import numpy as np
from cheartpy.vtk.types import VtkType

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pytools.arrays import Arr, SAny

MESHIO_2_CHEART_CONNECTIVITY: Mapping[VtkType, Mapping[int, int]] = {
    VtkType.LinLine: {0: 0, 1: 1},
    VtkType.QuadLine: {0: 0, 1: 2, 2: 1},
    VtkType.LinTriangle: {0: 0, 1: 1, 2: 2},
    VtkType.QuadTriangle: {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4},
    VtkType.LinQuadrilateral: {0: 0, 1: 1, 2: 3, 3: 2},
    VtkType.QuadQuadrilateral: {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 7, 6: 8, 7: 5, 8: 6},
    VtkType.LinTetrahedron: {0: 0, 1: 1, 2: 2, 3: 3},
    VtkType.QuadTetrahedron: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7, 8: 8, 9: 9},
    VtkType.LinHexahedron: {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 7, 7: 6},
    VtkType.QuadHexahedron: {
        0: 0,
        1: 1,
        2: 3,
        3: 2,
        4: 4,
        5: 5,
        6: 7,
        7: 6,
        8: 8,
        9: 11,
        10: 24,
        11: 9,
        12: 10,
        13: 16,
        14: 22,
        15: 17,
        16: 20,
        17: 26,
        18: 21,
        19: 19,
        20: 23,
        21: 18,
        22: 12,
        23: 15,
        24: 25,
        25: 13,
        26: 14,
    },
}
CHEART_2_MESHIO_CONNECTIVITY: Mapping[VtkType, Mapping[int, int]] = {
    element: {v: k for k, v in conn.items()}
    for element, conn in MESHIO_2_CHEART_CONNECTIVITY.items()
}


class CellBlock[D: SAny, F: np.floating, I: np.integer](abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int: ...
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def type(self) -> str: ...
    @property
    @abc.abstractmethod
    def data(self) -> list[int] | Arr[D, F] | Arr[D, I]: ...
    @property
    @abc.abstractmethod
    def dim(self) -> int: ...
    @property
    @abc.abstractmethod
    def tags(self) -> list[str]: ...


class MeshioMesh[D: SAny, F: np.floating, I: np.integer](abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def copy(self) -> MeshioMesh[D, F, I]: ...
    @abc.abstractmethod
    def write(self, path_or_buf: Path | TextIO, file_format: str | None) -> None: ...
    @property
    @abc.abstractmethod
    def points(self) -> Arr[D, F]: ...
    @points.setter
    @abc.abstractmethod
    def points(self, points: Arr[D, F]) -> None: ...
    @property
    @abc.abstractmethod
    def cells(self) -> Mapping[str, CellBlock[D, F, I]]: ...
    @property
    @abc.abstractmethod
    def point_data(self) -> Mapping[str, Arr[D, F]]: ...
    @property
    @abc.abstractmethod
    def cell_data(self) -> Mapping[str, Arr[D, F]]: ...
    @property
    @abc.abstractmethod
    def field_data(self) -> Mapping[str, Arr[D, F]]: ...
    @property
    @abc.abstractmethod
    def point_sets(self) -> Mapping[str, Arr[D, F]]: ...
    @property
    @abc.abstractmethod
    def cell_sets(self) -> Mapping[str, Arr[D, F]]: ...
