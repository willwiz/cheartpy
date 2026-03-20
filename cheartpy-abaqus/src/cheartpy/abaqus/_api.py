from typing import TYPE_CHECKING, NamedTuple, Unpack

import numpy as np
from cheartpy.io.api import chwrite_str_utf
from cheartpy.mesh.struct import CheartMesh
from pytools.logging import get_logger
from pytools.result import Err, Ok, Result

from .conversion import (
    ElemIntermediate,
    IndexUpdateMap,
    compile_abaqus_elements,
    compile_boundary_patches,
    compile_mask_data,
    compile_new_node_map,
    create_mesh_boundary,
    create_mesh_masks,
    create_mesh_space,
    create_mesh_topology,
)
from .reader import AbaqusMesh, import_abaqus_files

if TYPE_CHECKING:
    from pytools.arrays import DType

    from .parsing import AbaqusAPIKwargs


class _AbaqusData[F: np.floating, I: np.integer](NamedTuple):
    mesh: AbaqusMesh[F, I]
    body: ElemIntermediate[I]
    nmap: IndexUpdateMap


def _import_abaqus_data[F: np.floating, I: np.integer](
    *,
    ftype: DType[F] = np.float64,
    dtype: DType[I] = np.intp,
    **kwargs: Unpack[AbaqusAPIKwargs],
) -> Result[_AbaqusData[F, I]]:
    log = get_logger()
    match import_abaqus_files(*kwargs["files"], ftype=ftype, dtype=dtype):
        case Ok(abaqus): ...  # fmt: skip
        case Err(err):
            return Err(err)
    log.info(
        f"Imported {len(abaqus.nodes.v)} nodes.",
        f"Found {len(abaqus.elements)} element with types:",
        {k: elem.type for k, elem in abaqus.elements.items()},
        f"{len(abaqus.elsets)} were specified (name: nelem):",
        {k: len(elset.v) for k, elset in abaqus.elsets.items()},
        f"{len(abaqus.nset)} node sets were specified (name: nnodes):",
        {k: len(nset.v) for k, nset in abaqus.nset.items()},
    )
    match compile_abaqus_elements(abaqus, kwargs["topology"]):
        case Ok(body_elems): ...  # fmt: skip
        case Err(err):
            return Err(err)
    nmap = compile_new_node_map(body_elems)
    return Ok(_AbaqusData(mesh=abaqus, body=body_elems, nmap=nmap))


def _create_cheartmesh_from_abaqus_data[F: np.floating, I: np.integer](
    data: _AbaqusData[F, I],
    **kwargs: Unpack[AbaqusAPIKwargs],
) -> Result[CheartMesh[F, I]]:
    match compile_boundary_patches(data.mesh, data.body, kwargs.get("boundary")):
        case Ok(patches): ...  # fmt: skip
        case Err(err):
            return Err(err)
    space = create_mesh_space(data.mesh, data.nmap)
    match create_mesh_topology(data.body, data.nmap):
        case Ok(topology): ...  # fmt: skip
        case Err(err):
            return Err(err)
    match create_mesh_boundary(data.body, data.nmap, patches):
        case Ok(boundary): ...  # fmt: skip
        case Err(err):
            return Err(err)
    mesh = CheartMesh(space=space, top=topology, bnd=boundary)
    return Ok(mesh)


def create_cheartmesh_from_abaqus_api[F: np.floating, I: np.integer](
    *,
    ftype: DType[F] = np.float64,
    dtype: DType[I] = np.intp,
    **kwargs: Unpack[AbaqusAPIKwargs],
) -> Result[CheartMesh[F, I]]:
    get_logger(level=kwargs.get("log_level"))
    match _import_abaqus_data(**kwargs, ftype=ftype, dtype=dtype):
        case Ok(abaqus): ...  # fmt: skip
        case Err(err):
            return Err(err)
    match _create_cheartmesh_from_abaqus_data(abaqus, **kwargs):
        case Ok(mesh): ...  # fmt: skip
        case Err(err):
            return Err(err)
    masks = compile_mask_data(abaqus.mesh, kwargs.get("masks"))
    match create_mesh_masks(abaqus.nmap, masks):
        case Ok(mesh_masks): ...  # fmt: skip
        case Err(err):
            return Err(err)
    if (prefix := kwargs.get("prefix")) is not None:
        mesh.save(prefix)
    for k, v in mesh_masks.items():
        chwrite_str_utf(k, v[:, None])
    return Ok(mesh)
