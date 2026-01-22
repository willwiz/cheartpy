import dataclasses as dc
import enum
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

if TYPE_CHECKING:
    import numpy as np
    from pytools.arrays import A1, A2

__all__ = ["VTK_TYPE", "VtkElem", "VtkType"]


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


class _ShapeFunc(Protocol):
    def __call__[F: np.floating](self, pos: A1[F]) -> A1[F]: ...


class _ShapeFuncDeriv(Protocol):
    def __call__[F: np.floating](self, pos: A1[F]) -> A2[F]: ...


@dc.dataclass(slots=True, frozen=True)
class VtkElem:
    body: VtkType
    surf: VtkType | None
    connectivity: tuple[int, ...]
    nodes: A2[np.intc]
    ref: A2[np.float64]
    shape_func: _ShapeFunc
    shape_dfunc: _ShapeFuncDeriv
