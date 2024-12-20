__all__ = ["IVtkElementInterface", "get_element_type"]
import abc
from typing import TextIO, Protocol, ClassVar
from ..var_types import int_t, Arr


class IVtkElementInterface(Protocol):
    vtkelementid: ClassVar[int]
    vtksurfaceid: ClassVar[int | None]
    # connectivity: ClassVar[tuple[int, ...]]

    @staticmethod
    @abc.abstractmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None: ...


class VtkLinearLine(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 3
    vtksurfaceid: ClassVar[int | None] = None
    connectivity = (0, 1)

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for j in range(2):
            fout.write(" %i" % (elem[j] - 1))
        fout.write("\n")


class VtkQuadraticLine(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 21
    vtksurfaceid: ClassVar[int | None] = None
    connectivity = (0, 1, 2)

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for j in range(3):
            fout.write(" %i" % (elem[j] - 1))
        fout.write("\n")


class VtkBilinearTriangle(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 5
    vtksurfaceid: ClassVar[int | None] = 3
    connectivity = (0, 1, 2)

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for j in range(3):
            fout.write(" %i" % (elem[j] - 1))
        fout.write("\n")


class VtkBiquadraticTriangle(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 22
    vtksurfaceid: ClassVar[int | None] = 21
    connectivity = (0, 1, 2, 3, 5, 4)

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.write(" %i" % (elem[0] - 1))
        fout.write(" %i" % (elem[1] - 1))
        fout.write(" %i" % (elem[2] - 1))
        fout.write(" %i" % (elem[3] - 1))
        fout.write(" %i" % (elem[5] - 1))
        fout.write(" %i" % (elem[4] - 1))
        fout.write("\n")


class VtkBilinearQuadrilateral(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 9
    vtksurfaceid: ClassVar[int | None] = 3

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.write(" %i" % (elem[0] - 1))
        fout.write(" %i" % (elem[1] - 1))
        fout.write(" %i" % (elem[3] - 1))
        fout.write(" %i" % (elem[2] - 1))
        fout.write("\n")


class VtkTrilinearTetrahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 10
    vtksurfaceid: ClassVar[int | None] = 5

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for j in range(4):
            fout.write(" %i" % (elem[j] - 1))
        fout.write("\n")


class VtkBiquadraticQuadrilateral(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 28
    vtksurfaceid: ClassVar[int | None] = 21

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.write(" %i" % (elem[0] - 1))
        fout.write(" %i" % (elem[1] - 1))
        fout.write(" %i" % (elem[3] - 1))
        fout.write(" %i" % (elem[2] - 1))
        fout.write(" %i" % (elem[4] - 1))
        fout.write(" %i" % (elem[7] - 1))
        fout.write(" %i" % (elem[8] - 1))
        fout.write(" %i" % (elem[5] - 1))
        fout.write(" %i" % (elem[6] - 1))
        fout.write("\n")


class VtkTriquadraticTetrahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 24
    vtksurfaceid: ClassVar[int | None] = 22

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for j in range(10):
            if j == 6:
                fout.write(" %i" % (elem[5] - 1))
            elif j == 5:
                fout.write(" %i" % (elem[6] - 1))
            else:
                fout.write(" %i" % (elem[j] - 1))
        fout.write("\n")


class VtkTrilinearHexahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 12
    vtksurfaceid: ClassVar[int | None] = 9

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.write(" %i" % (elem[0] - 1))
        fout.write(" %i" % (elem[1] - 1))
        fout.write(" %i" % (elem[5] - 1))
        fout.write(" %i" % (elem[4] - 1))
        fout.write(" %i" % (elem[2] - 1))
        fout.write(" %i" % (elem[3] - 1))
        fout.write(" %i" % (elem[7] - 1))
        fout.write(" %i" % (elem[6] - 1))
        fout.write("\n")


class VtkTriquadraticHexahedron(IVtkElementInterface):
    vtkelementid: ClassVar[int] = 29
    vtksurfaceid: ClassVar[int | None] = 28

    @staticmethod
    def write(fout: TextIO, elem: Arr[tuple[int], int_t], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        fout.write(" %i" % (elem[0] - 1))
        fout.write(" %i" % (elem[1] - 1))
        fout.write(" %i" % (elem[5] - 1))
        fout.write(" %i" % (elem[4] - 1))
        fout.write(" %i" % (elem[2] - 1))
        fout.write(" %i" % (elem[3] - 1))
        fout.write(" %i" % (elem[7] - 1))
        fout.write(" %i" % (elem[6] - 1))
        fout.write(" %i" % (elem[8] - 1))
        fout.write(" %i" % (elem[15] - 1))
        fout.write(" %i" % (elem[22] - 1))
        fout.write(" %i" % (elem[13] - 1))
        fout.write(" %i" % (elem[12] - 1))
        fout.write(" %i" % (elem[21] - 1))
        fout.write(" %i" % (elem[26] - 1))
        fout.write(" %i" % (elem[19] - 1))
        fout.write(" %i" % (elem[9] - 1))
        fout.write(" %i" % (elem[11] - 1))
        fout.write(" %i" % (elem[25] - 1))
        fout.write(" %i" % (elem[23] - 1))
        fout.write(" %i" % (elem[16] - 1))
        fout.write(" %i" % (elem[18] - 1))
        fout.write(" %i" % (elem[10] - 1))
        fout.write(" %i" % (elem[24] - 1))
        fout.write(" %i" % (elem[14] - 1))
        fout.write(" %i" % (elem[20] - 1))
        fout.write(" %i" % (elem[17] - 1))
        fout.write("\n")


VtkTopologyElement = (
    type[VtkBilinearTriangle]
    | type[VtkBiquadraticTriangle]
    | type[VtkBilinearQuadrilateral]
    | type[VtkBiquadraticQuadrilateral]
    | type[VtkTrilinearTetrahedron]
    | type[VtkTriquadraticTetrahedron]
    | type[VtkTrilinearHexahedron]
    | type[VtkTriquadraticHexahedron]
)
VtkBoundaryElement = (
    type[VtkLinearLine]
    | type[VtkQuadraticLine]
    | type[VtkBilinearTriangle]
    | type[VtkBiquadraticTriangle]
    | type[VtkBilinearQuadrilateral]
    | type[VtkBiquadraticQuadrilateral]
)


def get_element_type(
    nnodes: int, boundary: str | None
) -> tuple[type[IVtkElementInterface], type[IVtkElementInterface]]:
    if boundary is None:
        nbnd = None
    else:
        with open(boundary, "r") as f:
            _ = f.readline()
            nbnd = len(f.readline().strip().split())
    match nnodes:
        case 3:
            return VtkBilinearTriangle, VtkLinearLine
        case 6:
            return VtkBiquadraticTriangle, VtkQuadraticLine
        case 4:
            if nbnd == 4:
                return VtkBilinearQuadrilateral, VtkLinearLine
            elif nbnd == 5:
                return VtkTrilinearTetrahedron, VtkBilinearTriangle
            raise ValueError(
                f"Bilinear quadrilateral / Trilinear tetrahedron detected but ambiguous"
            )
        case 9:
            return VtkBiquadraticQuadrilateral, VtkQuadraticLine
        case 10:
            return VtkTriquadraticTetrahedron, VtkBiquadraticTriangle
        case 8:
            return VtkTrilinearHexahedron, VtkBilinearQuadrilateral
        case 27:
            return VtkTriquadraticHexahedron, VtkBiquadraticQuadrilateral
        case _:
            raise ValueError(
                f"Cannot determine element type from {nnodes} and {
                    boundary}, perhaps not implemented."
            )
