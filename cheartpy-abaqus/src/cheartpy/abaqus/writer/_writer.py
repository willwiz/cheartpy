import textwrap
from typing import TYPE_CHECKING, TextIO, Unpack

import numpy as np
from cheartpy.elem_interfaces import convert_vtk_to_abaqus
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cheartpy.mesh import CheartMesh, CheartMeshPatch
    from pytools.arrays import A1

    from ._types import AbaqusWriterKwargs


def validate_element_sets[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], elset: Mapping[str, A1[np.integer]] | None
) -> Result[None]:
    if elset is None:
        return Ok(None)
    for name, elements in elset.items():
        if all(0 <= element < mesh.top.n for element in elements):
            continue
        msg = (
            f"Element set '{name}' contains invalid element IDs. Valid range is 1 to {mesh.top.n}."
        )
        return Err(ValueError(msg))
    return Ok(None)


def validate_node_sets[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], nset: Mapping[str, A1[np.integer]] | None
) -> Result[None]:
    if nset is None:
        return Ok(None)
    for name, nodes in nset.items():
        if all(0 <= node < mesh.space.n for node in nodes):
            continue
        msg = f"Node set '{name}' contains invalid node IDs. Valid range is 1 to {mesh.space.n}."
        return Err(ValueError(msg))
    return Ok(None)


def write_header(f: TextIO, header: Sequence[str] | None) -> None:
    if header is None:
        return
    f.write("*HEADING\n")
    wrapped_header = (
        line for head in header for wrapped in textwrap.wrap(head, width=80) for line in wrapped
    )
    f.write("\n".join(f"{'** '}{line}\n" for line in wrapped_header))


def write_nodes[F: np.floating, I: np.integer](f: TextIO, mesh: CheartMesh[F, I]) -> Result[None]:
    f.write("*NODE\n")
    nnodes, dim = mesh.space.v.shape
    dtype = [("index", np.intp), *[(f"x_{i}", np.float64) for i in range(dim)]]
    export_array = np.empty(nnodes, dtype=dtype)
    export_array["index"] = np.arange(1, nnodes + 1, dtype=np.intp)
    for i in range(dim):
        export_array[f"x_{i}"] = mesh.space.v[:, i]
    fmt = ["%d"] + ["%.16g"] * dim
    np.savetxt(f, export_array, fmt=fmt, delimiter=", ")
    return Ok(None)


def write_volume[F: np.floating, I: np.integer](f: TextIO, mesh: CheartMesh[F, I]) -> Result[int]:
    element_type = convert_vtk_to_abaqus(mesh.top.TYPE)
    header = f"*ELEMENT, TYPE={element_type!s}, ELSET=Volume1\n"
    nelem, nnode = mesh.top.v.shape
    dtype = [("index", np.intp), *[(f"i_{i}", np.intp) for i in range(nnode)]]
    export_array = np.empty(nelem, dtype=dtype)
    export_array["index"] = np.arange(1, nelem + 1, dtype=np.intp)
    for i in range(nnode):
        export_array[f"i_{i}"] = mesh.top.v[:, i] + 1
    f.write(header)
    fmt = ["%d"] + ["%d"] * nnode
    np.savetxt(f, export_array, fmt=fmt, delimiter=", ")
    return Ok(nelem)


def write_surface[F: np.floating, I: np.integer](
    f: TextIO, idx: int, patch: CheartMeshPatch[I], current_elem: int
) -> Result[int]:
    element_type = convert_vtk_to_abaqus(patch.TYPE)
    header = f"*ELEMENT, TYPE={element_type!s}, ELSET=Surface{idx}\n"
    nelem, nnode = patch.v.shape
    dtype = [("index", np.intp), *[(f"i_{i}", np.intp) for i in range(nnode)]]
    export_array = np.empty(nelem, dtype=dtype)
    export_array["index"] = np.arange(current_elem + 1, current_elem + nelem + 1, dtype=np.intp)
    for i in range(nnode):
        export_array[f"i_{i}"] = patch.v[:, i] + 1
    f.write(header)
    fmt = ["%d"] + ["%d"] * nnode
    np.savetxt(f, export_array, fmt=fmt, delimiter=", ")
    return Ok(current_elem + nelem)


def write_surfaces[F: np.floating, I: np.integer](
    f: TextIO, mesh: CheartMesh[F, I], current_elem: int
) -> None:
    if mesh.bnd is None:
        return
    for idx, patch in mesh.bnd.v.items():
        match write_surface(f, idx, patch, current_elem):
            case Ok(new_current_elem):
                current_elem = new_current_elem
            case Err(e):
                raise e


def write_element_sets[F: np.floating, I: np.integer](
    f: TextIO, elset: Mapping[str, A1[np.integer]] | None
) -> None:
    if elset is None:
        return
    for name, elements in elset.items():
        f.write(f"*ELSET, ELSET={name}\n")
        for i in range(0, len(elements), 16):
            line = ", ".join(str(e + 1) for e in elements[i : i + 16])
            f.write(f"{line}\n")


def write_node_sets[F: np.floating, I: np.integer](
    f: TextIO, nset: Mapping[str, A1[np.integer]] | None
) -> None:
    if nset is None:
        return
    for name, nodes in nset.items():
        f.write(f"*NSET, NSET={name}\n")
        for i in range(0, len(nodes), 16):
            line = ", ".join(str(n + 1) for n in nodes[i : i + 16])
            f.write(f"{line}\n")


def write_loop[F: np.floating, I: np.integer](
    f: TextIO, mesh: CheartMesh[F, I], **kwargs: Unpack[AbaqusWriterKwargs]
) -> Result[None]:
    """Run all steps for write_inp_from_cheart."""
    write_header(f, kwargs.get("header"))
    match write_nodes(f, mesh):
        case Ok(_): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match write_volume(f, mesh):
        case Ok(current_elem): ...  # fmt: skip
        case Err(e):
            return Err(e)
    write_surfaces(f, mesh, current_elem)
    write_element_sets(f, kwargs.get("elset"))
    write_node_sets(f, kwargs.get("nset"))
    return Ok(None)


def write_inp_from_cheart[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], filename: Path, **kwargs: Unpack[AbaqusWriterKwargs]
) -> Result[None]:
    """Write a Cheart mesh to an Abaqus .inp file.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The Cheart mesh to write.
    filename : str
        The name of the output .inp file.
    header : Sequence[str], optional
        The header lines to write at the top of the .inp file.
    elset : Mapping[str, Sequence[int]], optional
        A mapping of element set names to lists of element IDs.
    nset : Mapping[str, Sequence[int]], optional
        A mapping of node set names to lists of node IDs.

    Returns
    -------
    Result[None]
        Ok(None) if successful, or Err(ValueError) if there was an error validating the element or
        node sets.

    """
    match validate_node_sets(mesh, kwargs.get("nset")):
        case Ok(_): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match validate_element_sets(mesh, kwargs.get("elset")):
        case Ok(_): ...  # fmt: skip
        case Err(e):
            return Err(e)
    with filename.open("w") as f:
        return write_loop(f, mesh, **kwargs).next()
