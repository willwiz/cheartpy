from __future__ import annotations

__all__ = ["get_vtk_elem", "guess_elem_type_from_dim"]

from typing import TYPE_CHECKING

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
    from cheartpy.vtk.trait import VtkElem


def guess_elem_type_from_dim(edim: int, bdim: int | None) -> VtkElem:
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
            raise ValueError(msg)
        case _:
            msg = f"Unsupported element dimensions: edim={edim}, bdim={bdim}"
            raise ValueError(msg)
    return elem
