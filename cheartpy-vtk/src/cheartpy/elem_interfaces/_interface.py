from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict, overload

from pytools.result import Err, Ok, Result

from ._abaqus import get_abaqus_elem_nodes
from ._cheart import get_cheart_elem_nodes
from ._types import (
    AbaqusEnum,
    CheartEnum,
    ElemEnum,
    ElemType,
    GmshEnum,
    NodeOrder,
    VtkElemShape,
    VtkEnum,
)
from ._vtk import get_vtk_elem_nodes

if TYPE_CHECKING:
    from collections.abc import Mapping


CHEART_2_VTK = {
    CheartEnum.VERTEX: VtkEnum.VERTEX,
    CheartEnum.LINE1: VtkEnum.LINE1,
    CheartEnum.TRIANGLE1: VtkEnum.TRIANGLE1,
    CheartEnum.QUADRILATERAL1: VtkEnum.QUADRILATERAL1,
    CheartEnum.TETRAHEDRON1: VtkEnum.TETRAHEDRON1,
    CheartEnum.HEXAHEDRON1: VtkEnum.HEXAHEDRON1,
    CheartEnum.LINE2: VtkEnum.LINE2,
    CheartEnum.TRIANGLE2: VtkEnum.TRIANGLE2,
    CheartEnum.QUADRILATERAL2: VtkEnum.QUADRILATERAL2,
    CheartEnum.TETRAHEDRON2: VtkEnum.TETRAHEDRON2,
    CheartEnum.HEXAHEDRON2: VtkEnum.HEXAHEDRON2,
}
VTK_2_CHEART = {v: k for k, v in CHEART_2_VTK.items()}

ABAQUS_2_VTK = {
    AbaqusEnum.VERTEX: VtkEnum.VERTEX,
    AbaqusEnum.S3R: VtkEnum.TRIANGLE1,
    AbaqusEnum.CPEG6: VtkEnum.TRIANGLE2,
    AbaqusEnum.LINE1: VtkEnum.LINE1,
    AbaqusEnum.LINE2: VtkEnum.LINE2,
    AbaqusEnum.TRIANGLE1: VtkEnum.TRIANGLE1,
    AbaqusEnum.TRIANGLE2: VtkEnum.TRIANGLE2,
    AbaqusEnum.QUADRILATERAL1: VtkEnum.QUADRILATERAL1,
    AbaqusEnum.QUADRILATERAL2: VtkEnum.QUADRILATERAL2,
    AbaqusEnum.TETRAHEDRON1: VtkEnum.TETRAHEDRON1,
    AbaqusEnum.TETRAHEDRON2: VtkEnum.TETRAHEDRON2,
    AbaqusEnum.HEXAHEDRON1: VtkEnum.HEXAHEDRON1,
    AbaqusEnum.HEXAHEDRON2: VtkEnum.HEXAHEDRON2,
}
VTK_2_ABAQUS = {v: k for k, v in ABAQUS_2_VTK.items()}
GMSH_2_VTK = {
    GmshEnum.VERTEX: VtkEnum.VERTEX,
    GmshEnum.LINE1: VtkEnum.LINE1,
    GmshEnum.LINE2: VtkEnum.LINE2,
    GmshEnum.TRIANGLE1: VtkEnum.TRIANGLE1,
    GmshEnum.TRIANGLE2: VtkEnum.TRIANGLE2,
    GmshEnum.QUADRILATERAL1: VtkEnum.QUADRILATERAL1,
    GmshEnum.QUADRILATERAL2: VtkEnum.QUADRILATERAL2,
    GmshEnum.TETRAHEDRON1: VtkEnum.TETRAHEDRON1,
    GmshEnum.TETRAHEDRON2: VtkEnum.TETRAHEDRON2,
    GmshEnum.HEXAHEDRON1: VtkEnum.HEXAHEDRON1,
    GmshEnum.HEXAHEDRON2: VtkEnum.HEXAHEDRON2,
}
VTK_2_GMSH = {v: k for k, v in GMSH_2_VTK.items()}

ABAQUS_2_CHEART = {k: VTK_2_CHEART[v] for k, v in ABAQUS_2_VTK.items()}
CHEART_2_ABAQUS = {v: k for k, v in ABAQUS_2_CHEART.items()}
GMSH_2_CHEART = {k: VTK_2_CHEART[v] for k, v in GMSH_2_VTK.items()}
CHEART_2_GMSH = {v: k for k, v in GMSH_2_CHEART.items()}
ABAQUS_2_GMSH = {k: VTK_2_GMSH[v] for k, v in ABAQUS_2_VTK.items()}
GMSH_2_ABAQUS = {v: k for k, v in ABAQUS_2_GMSH.items()}


class __CheartConverter(TypedDict, total=True):
    Cheart: Mapping[CheartEnum, CheartEnum]
    Vtk: Mapping[CheartEnum, VtkEnum]
    Abaqus: Mapping[CheartEnum, AbaqusEnum]
    Gmsh: Mapping[CheartEnum, GmshEnum]


class __VtkConverter(TypedDict, total=True):
    Cheart: Mapping[VtkEnum, CheartEnum]
    Vtk: Mapping[VtkEnum, VtkEnum]
    Abaqus: Mapping[VtkEnum, AbaqusEnum]
    Gmsh: Mapping[VtkEnum, GmshEnum]


class __AbaqusConverter(TypedDict, total=True):
    Cheart: Mapping[AbaqusEnum, CheartEnum]
    Vtk: Mapping[AbaqusEnum, VtkEnum]
    Abaqus: Mapping[AbaqusEnum, AbaqusEnum]
    Gmsh: Mapping[AbaqusEnum, GmshEnum]


class __GmshConverter(TypedDict, total=True):
    Cheart: Mapping[GmshEnum, CheartEnum]
    Vtk: Mapping[GmshEnum, VtkEnum]
    Abaqus: Mapping[GmshEnum, AbaqusEnum]
    Gmsh: Mapping[GmshEnum, GmshEnum]


class _NullConverter[T](dict[T, T]):
    def __getitem__(self, key: T) -> T:
        return key


_CheartConvert: __CheartConverter = {
    "Cheart": _NullConverter[CheartEnum](),
    "Vtk": CHEART_2_VTK,
    "Abaqus": CHEART_2_ABAQUS,
    "Gmsh": CHEART_2_GMSH,
}
_VtkConvert: __VtkConverter = {
    "Cheart": VTK_2_CHEART,
    "Vtk": _NullConverter[VtkEnum](),
    "Abaqus": VTK_2_ABAQUS,
    "Gmsh": VTK_2_GMSH,
}
_AbaqusConvert: __AbaqusConverter = {
    "Cheart": ABAQUS_2_CHEART,
    "Vtk": ABAQUS_2_VTK,
    "Abaqus": _NullConverter[AbaqusEnum](),
    "Gmsh": ABAQUS_2_GMSH,
}
_GmshConvert: __GmshConverter = {
    "Cheart": GMSH_2_CHEART,
    "Vtk": GMSH_2_VTK,
    "Abaqus": GMSH_2_ABAQUS,
    "Gmsh": _NullConverter[GmshEnum](),
}

_Enum: Mapping[ElemType, type[ElemEnum]] = {
    "Cheart": CheartEnum,
    "Vtk": VtkEnum,
    "Abaqus": AbaqusEnum,
    "Gmsh": GmshEnum,
}


@overload
def convert_element_type(input_type: ElemEnum, target: Literal["Cheart"]) -> CheartEnum: ...
@overload
def convert_element_type(input_type: ElemEnum, target: Literal["Vtk"]) -> VtkEnum: ...
@overload
def convert_element_type(input_type: ElemEnum, target: Literal["Abaqus"]) -> AbaqusEnum: ...
@overload
def convert_element_type(input_type: ElemEnum, target: Literal["Gmsh"]) -> GmshEnum: ...
def convert_element_type(input_type: ElemEnum, target: ElemType) -> ElemEnum:
    """Convert an element type from one enum to another.

    Parameters
    ----------
    input_type : CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The input element enum.
    target : Literal["Cheart", "Vtk", "Abaqus", "Gmsh"]
        The target element class string.

    Returns
    -------
    CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The target enum corresponding to the target string.

    """
    match input_type:
        case CheartEnum():
            return _CheartConvert[target][input_type]
        case VtkEnum():
            return _VtkConvert[target][input_type]
        case AbaqusEnum():
            return _AbaqusConvert[target][input_type]
        case GmshEnum():
            return _GmshConvert[target][input_type]


def get_node_order(elem: ElemEnum) -> NodeOrder:
    """Return the node order mapping for the given element type.

    Parameters
    ----------
    elem : CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The element type enum.

    Returns
    -------
    Mapping[int, tuple[int, int, int]]
        The node order mapping for the given element type.

    """
    match elem:
        case VtkEnum():
            return get_vtk_elem_nodes(elem)
        case CheartEnum():
            return get_cheart_elem_nodes(elem)
        case AbaqusEnum():
            return get_abaqus_elem_nodes(elem)
        case GmshEnum():
            return get_vtk_elem_nodes(GMSH_2_VTK[elem])


_VtkBoundaryElement: dict[VtkEnum, VtkEnum] = {
    VtkEnum.LINE1: VtkEnum.VERTEX,
    VtkEnum.LINE2: VtkEnum.VERTEX,
    VtkEnum.TRIANGLE1: VtkEnum.LINE1,
    VtkEnum.TRIANGLE2: VtkEnum.LINE2,
    VtkEnum.QUADRILATERAL1: VtkEnum.LINE1,
    VtkEnum.QUADRILATERAL2: VtkEnum.LINE2,
    VtkEnum.TETRAHEDRON1: VtkEnum.TRIANGLE1,
    VtkEnum.TETRAHEDRON2: VtkEnum.TRIANGLE2,
    VtkEnum.HEXAHEDRON1: VtkEnum.QUADRILATERAL1,
    VtkEnum.HEXAHEDRON2: VtkEnum.QUADRILATERAL2,
}


def _get_vtk_boundary_element(elem: VtkEnum) -> VtkEnum:
    match _VtkBoundaryElement.get(elem):
        case None:
            msg = f"{elem} cannot have a boundary element."
            raise ValueError(msg)
        case boundary_elem:
            return boundary_elem


def get_boundary_element[T: ElemEnum](elem: T) -> T:
    """Return the boundary element type for the given element type.

    Parameters
    ----------
    elem : CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The element type enum.

    Returns
    -------
    CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The boundary element type enum.

    Raises
    ------
    ValueError
        If the element type does not have a boundary element.

    """
    match elem:
        case VtkEnum():
            return _get_vtk_boundary_element(elem)
        case CheartEnum():
            vtk_elem = CHEART_2_VTK[elem]
            vtk_boundary_elem = _get_vtk_boundary_element(vtk_elem)
            return VTK_2_CHEART[vtk_boundary_elem]
        case AbaqusEnum():
            vtk_elem = ABAQUS_2_VTK[elem]
            vtk_boundary_elem = _get_vtk_boundary_element(vtk_elem)
            return VTK_2_ABAQUS[vtk_boundary_elem]
        case GmshEnum():
            vtk_elem = GMSH_2_VTK[elem]
            vtk_boundary_elem = _get_vtk_boundary_element(vtk_elem)
            return VTK_2_GMSH[vtk_boundary_elem]


_DIM_TO_VTK_ELEM: Mapping[tuple[int, int | None], VtkEnum | None] = {
    (3, 2): VtkEnum.TRIANGLE1,
    (3, None): VtkEnum.TRIANGLE1,
    (6, 3): VtkEnum.TRIANGLE2,
    (6, None): VtkEnum.TRIANGLE2,
    (4, 2): VtkEnum.QUADRILATERAL1,
    (4, None): None,
    (9, 3): VtkEnum.QUADRILATERAL2,
    (9, None): VtkEnum.QUADRILATERAL2,
    (4, 3): VtkEnum.TETRAHEDRON1,
    (10, 6): VtkEnum.TETRAHEDRON2,
    (10, None): VtkEnum.TETRAHEDRON2,
    (8, 4): VtkEnum.HEXAHEDRON1,
    (8, None): VtkEnum.HEXAHEDRON1,
    (27, 9): VtkEnum.HEXAHEDRON2,
    (27, None): VtkEnum.HEXAHEDRON2,
}

_ERR_MSG = {
    4: "(size = 4) Cannot distinguish linear quadrilateral and linear tets, need boundary dim",
}


@overload
def guess_element_from_dim(
    edim: int, bdim: int | None, target: Literal["Gmsh"]
) -> Result[GmshEnum]: ...
@overload
def guess_element_from_dim(
    edim: int, bdim: int | None, target: Literal["Abaqus"]
) -> Result[AbaqusEnum]: ...
@overload
def guess_element_from_dim(
    edim: int, bdim: int | None, target: Literal["Vtk"]
) -> Result[VtkEnum]: ...
@overload
def guess_element_from_dim(
    edim: int, bdim: int | None, target: Literal["Cheart"]
) -> Result[CheartEnum]: ...
def guess_element_from_dim(edim: int, bdim: int | None, target: ElemType) -> Result[ElemEnum]:
    match _DIM_TO_VTK_ELEM.get((edim, bdim)):
        case VtkEnum() as vtk_type:
            ...
        case None:
            msg = _ERR_MSG.get(edim, f"Unsupported element dimensions: edim={edim}, bdim={bdim}")
            return Err(ValueError(msg))
    match target:
        case "Abaqus":
            return Ok(convert_element_type(vtk_type, "Abaqus"))
        case "Cheart":
            return Ok(convert_element_type(vtk_type, "Cheart"))
        case "Gmsh":
            return Ok(convert_element_type(vtk_type, "Gmsh"))
        case "Vtk":
            return Ok(vtk_type)


_VtkEnumCategory: dict[tuple[VtkElemShape, int], VtkEnum] = {
    ("Line", 1): VtkEnum.LINE1,
    ("Triangle", 1): VtkEnum.TRIANGLE1,
    ("Quadrilateral", 1): VtkEnum.QUADRILATERAL1,
    ("Tetrahedron", 1): VtkEnum.TETRAHEDRON1,
    ("Hexahedron", 1): VtkEnum.HEXAHEDRON1,
    ("Line", 2): VtkEnum.LINE2,
    ("Triangle", 2): VtkEnum.TRIANGLE2,
    ("Quadrilateral", 2): VtkEnum.QUADRILATERAL2,
    ("Tetrahedron", 2): VtkEnum.TETRAHEDRON2,
    ("Hexahedron", 2): VtkEnum.HEXAHEDRON2,
}


def get_element_enum_from_polyorder(
    elem: VtkElemShape, order: int, target: ElemType
) -> Result[ElemEnum]:
    """Get the element enum from the element shape and polynomial order.

    Parameters
    ----------
    elem : VtkElemShape
        The element shape.
    order : int
        The polynomial order.
    target : Literal["Cheart", "Vtk", "Abaqus", "Gmsh"]
        The target element class string.

    Returns
    -------
    Result[CheartEnum] | Result[VtkEnum] | Result[AbaqusEnum] | Result[GmshEnum]
        The element enum corresponding to the target string.

    """
    match _VtkEnumCategory.get((elem, order)):
        case VtkEnum() as vtk_elem: ...  # fmt: skip
        case None:
            msg = f"Unsupported element shape and polynomial order: elem={elem}, order={order}"
            return Err(ValueError(msg))
    match target:
        case "Cheart":
            return Ok(convert_element_type(vtk_elem, "Cheart"))
        case "Vtk":
            return Ok(vtk_elem)
        case "Abaqus":
            return Ok(convert_element_type(vtk_elem, "Abaqus"))
        case "Gmsh":
            return Ok(convert_element_type(vtk_elem, "Gmsh"))


def get_element_shape(elem: ElemEnum) -> VtkElemShape:
    """Get the element shape from the element enum.

    Parameters
    ----------
    elem : CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The element type enum.

    Returns
    -------
    VtkElemShape
        The element shape.

    """
    match elem:
        case VtkEnum():
            return elem.shape
        case CheartEnum():
            return CHEART_2_VTK[elem].shape
        case AbaqusEnum():
            return ABAQUS_2_VTK[elem].shape
        case GmshEnum():
            return GMSH_2_VTK[elem].shape


def get_element_order(elem: ElemEnum) -> int:
    """Get the polynomial order from the element enum.

    Parameters
    ----------
    elem : CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The element type enum.

    Returns
    -------
    int
        The polynomial order.

    """
    match elem:
        case VtkEnum():
            return elem.order
        case CheartEnum():
            return CHEART_2_VTK[elem].order
        case AbaqusEnum():
            return ABAQUS_2_VTK[elem].order
        case GmshEnum():
            return GMSH_2_VTK[elem].order
