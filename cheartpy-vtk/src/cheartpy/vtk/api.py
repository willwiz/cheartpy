from typing import TYPE_CHECKING

from pytools.result import Err, Ok

from ._elements import (
    dlagrange_1,
    dlagrange_2,
    dtri_lagrange_1,
    dtri_lagrange_2,
    lagrange_1,
    lagrange_2,
    tri_lagrange_1,
    tri_lagrange_2,
)
from .struct import (
    VTKHEXAHEDRON1,
    VTKHEXAHEDRON2,
    VTKQUADRILATERAL1,
    VTKQUADRILATERAL2,
    VTKTETRAHEDRON1,
    VTKTRIANGLE1,
    VTKTRIANGLE2,
    get_vtk_elem,
)

if TYPE_CHECKING:
    from ._elements import VtkElem

__all__ = [
    "dlagrange_1",
    "dlagrange_2",
    "dtri_lagrange_1",
    "dtri_lagrange_2",
    "get_vtk_elem",
    "guess_elem_type_from_dim",
    "lagrange_1",
    "lagrange_2",
    "tri_lagrange_1",
    "tri_lagrange_2",
]


def guess_elem_type_from_dim(edim: int, bdim: int | None) -> Ok[VtkElem] | Err:
    match edim, bdim:
        case 3, 2 | None:
            elem = VTKTRIANGLE1
        case 6, 3 | None:
            elem = VTKTRIANGLE2
        case 4, 2:
            elem = VTKQUADRILATERAL1
        case 9, 3 | None:
            elem = VTKQUADRILATERAL2
        case 4, 3:
            elem = VTKTETRAHEDRON1
        case 10, 6 | None:
            elem = VTKQUADRILATERAL2
        case 8, 4 | None:
            elem = VTKHEXAHEDRON1
        case 27, 9 | None:
            elem = VTKHEXAHEDRON2
        case 4, None:
            msg = (
                "Cannot detect between Bilinear quadrilateral and Trilinear tetrahedron,"
                "need boundary dim"
            )
            return Err(ValueError(msg))
        case _:
            msg = f"Unsupported element dimensions: edim={edim}, bdim={bdim}"
            return Err(ValueError(msg))
    return Ok(elem)
