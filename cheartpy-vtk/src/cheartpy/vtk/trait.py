import dataclasses as dc

__all__ = ["VTK_TYPE", "VtkElem", "VtkType"]
import enum
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from pytools.arrays import A1, A2


class _Vtk(NamedTuple):
    name: str
    idx: int


class VtkType(enum.Enum):
    LinLine = _Vtk("VtkLinearLine", 3)
    LinTriangle = _Vtk("VtkLinearTriangle", 5)
    LinQuadrilateral = _Vtk("VtkLinearQuadrilateral", 9)
    LinTetrahedron = _Vtk("VtkLinearTetrahedron", 10)
    LinHexahedron = _Vtk("VtkLinearHexahedron", 12)
    QuadLine = _Vtk("VtkQuadraticLine", 21)
    QuadTriangle = _Vtk("VtkQuadraticTriangle", 22)
    QuadQuadrilateral = _Vtk("VtkQuadraticQuadrilateral", 28)
    QuadTetrahedron = _Vtk("VtkQuadraticTetrahedron", 24)
    QuadHexahedron = _Vtk("VtkQuadraticHexahedron", 29)


type VTK_TYPE = Literal[
    "LinLine",
    "LinTriangle",
    "LinQuadrilateral",
    "LinTetrahedron",
    "LinHexahedron",
    "QuadLine",
    "QuadTriangle",
    "QuadQuadrilateral",
    "QuadTetrahedron",
    "QuadHexahedron",
]


@dc.dataclass(slots=True, frozen=True)
class VtkElem:
    body: VtkType
    surf: VtkType | None
    connectivity: tuple[int, ...]
    nodes: A2[np.intc]
    ref: A2[np.float64]
    shape_func: Callable[[A1[np.floating]], A1[np.float64]]
    shape_dfunc: Callable[[A1[np.floating]], A2[np.float64]]
