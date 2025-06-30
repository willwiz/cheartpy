__all__ = ["IVtkElementInterface", "get_element_type"]
import abc
from pathlib import Path
from typing import ClassVar, Protocol, TextIO
from warnings import warn

import numpy as np
from arraystubs import Arr1


class IVtkElementInterface(Protocol):
    vtkelementid: ClassVar[int]
    vtksurfaceid: ClassVar[int | None]
    connectivity: ClassVar[tuple[int, ...]]

    @staticmethod
    @abc.abstractmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None: ...


class VtkLinearLine(IVtkElementInterface):
    vtkelementid = 3
    vtksurfaceid = None
    connectivity = (0, 1)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[j] - 1:d}" for j in VtkLinearLine.connectivity)
        fout.write("\n")


class VtkQuadraticLine(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 21
    vtksurfaceid: ClassVar[int | None] = None
    connectivity = (0, 1, 2)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[j] - 1:d}" for j in VtkQuadraticLine.connectivity)
        fout.write("\n")


class VtkBilinearTriangle(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 5
    vtksurfaceid: ClassVar[int | None] = 3
    connectivity = (0, 1, 2)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[j] - 1:d}" for j in [VtkBilinearTriangle.connectivity])
        fout.write("\n")


class VtkBiquadraticTriangle(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 22
    vtksurfaceid: ClassVar[int | None] = 21
    connectivity = (0, 1, 2, 3, 5, 4)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[i] - 1:d}" for i in VtkBiquadraticTriangle.connectivity)
        fout.write("\n")


class VtkBilinearQuadrilateral(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 9
    vtksurfaceid: ClassVar[int | None] = 3
    connectivity = (0, 1, 3, 2)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[i] - 1:d}" for i in VtkBilinearQuadrilateral.connectivity)
        fout.write("\n")


class VtkTrilinearTetrahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 10
    vtksurfaceid: ClassVar[int | None] = 5
    connectivity = (0, 1, 2, 3)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[j] - 1:d}" for j in VtkTrilinearTetrahedron.connectivity)
        fout.write("\n")


class VtkBiquadraticQuadrilateral(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 28
    vtksurfaceid: ClassVar[int | None] = 21
    connectivity = (0, 1, 3, 2, 4, 7, 8, 5, 6)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[i] - 1:d}" for i in VtkBiquadraticQuadrilateral.connectivity)
        fout.write("\n")


class VtkTriquadraticTetrahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 24
    vtksurfaceid: ClassVar[int | None] = 22
    connectivity = (
        0, 1, 2, 3, 4, 6, 5, 7, 8, 9,
    )  # fmt: skip

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[i] - 1:d}" for i in VtkTriquadraticTetrahedron.connectivity)
        fout.write("\n")


class VtkTrilinearHexahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 12
    vtksurfaceid: ClassVar[int | None] = 9
    connectivity = (0, 1, 5, 4, 2, 3, 7, 6)

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[i] - 1:d}" for i in VtkTrilinearHexahedron.connectivity)
        fout.write("\n")


class VtkTriquadraticHexahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 29
    vtksurfaceid: ClassVar[int | None] = 28
    connectivity = (
        0,  1,  5,  4,  2,  3,  7, 6,  8,  15,
        22, 13, 12, 21, 26, 19, 9, 11, 25, 23,
        16, 18, 10, 24, 14, 20, 17,
    )  # fmt: skip

    @staticmethod
    def write[T: np.integer](fout: TextIO, elem: Arr1[T], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.writelines(f" {elem[i] - 1:d}" for i in VtkTriquadraticHexahedron.connectivity)
        fout.write("\n")


type VtkTopologyElement = type[
    VtkBilinearTriangle
    | VtkBiquadraticTriangle
    | VtkBilinearQuadrilateral
    | VtkBiquadraticQuadrilateral
    | VtkTrilinearTetrahedron
    | VtkTriquadraticTetrahedron
    | VtkTrilinearHexahedron
    | VtkTriquadraticHexahedron
]
type VtkBoundaryElement = type[
    VtkLinearLine
    | VtkQuadraticLine
    | VtkBilinearTriangle
    | VtkBiquadraticTriangle
    | VtkBilinearQuadrilateral
    | VtkBiquadraticQuadrilateral
]


def get_element_type_from_nodes(
    nnodes: int,
    nbnd: int | None,
) -> tuple[VtkTopologyElement, VtkBoundaryElement]:
    match nnodes, nbnd:
        case 3, _:
            vtkelem = VtkBilinearTriangle, VtkLinearLine
        case 6, _:
            vtkelem = VtkBiquadraticTriangle, VtkQuadraticLine
        case 4, 4:
            vtkelem = VtkBilinearQuadrilateral, VtkLinearLine
        case 4, 5:
            vtkelem = VtkTrilinearTetrahedron, VtkBilinearTriangle
        case 4, None:
            msg = (
                "Bilinear quadrilateral / Trilinear tetrahedron detected"
                "It's ambiguous without boundary element size. Quadrilateral assumed."
            )
            warn(msg, stacklevel=2)
            vtkelem = VtkBilinearQuadrilateral, VtkLinearLine
        case 9, _:
            vtkelem = VtkBiquadraticQuadrilateral, VtkQuadraticLine
        case 10, _:
            vtkelem = VtkTriquadraticTetrahedron, VtkBiquadraticTriangle
        case 8, _:
            vtkelem = VtkTrilinearHexahedron, VtkBilinearQuadrilateral
        case 27, _:
            vtkelem = VtkTriquadraticHexahedron, VtkBiquadraticQuadrilateral
        case _:
            msg = (
                f"Cannot determine element type from {nnodes} nodes and "
                f"{nbnd}, perhaps not implemented."
            )
            raise ValueError(msg)
    return vtkelem


def get_element_type(
    nnodes: int,
    boundary: str | None,
) -> tuple[type[IVtkElementInterface], type[IVtkElementInterface]]:
    if boundary is None:
        nbnd = None
    else:
        with Path(boundary).open("r") as f:
            _ = next(f)  # skip header
            nbnd = len(next(f).strip().split())
    return get_element_type_from_nodes(nnodes, nbnd)
