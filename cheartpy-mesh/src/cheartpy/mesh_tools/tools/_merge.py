from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple, TypeIs

import numpy as np
from cheartpy.elem_interfaces import CheartEnum, get_element_size
from pytools.result import Err, Ok, Result

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

from ._types import IndexPermutation, MergedMesh
from ._validation import create_index_permutation

if TYPE_CHECKING:
    from pytools.arrays import A2, DType


class _MeshInformation[F: np.floating, I: np.integer](NamedTuple):
    ftype: np.dtype[F]
    itype: np.dtype[I]
    dim: int
    elem: CheartEnum


def _check_mesh_information[F: np.floating, I: np.integer](
    meshes: Mapping[int, CheartMesh[F, I]],
) -> Result[_MeshInformation[F, I]]:
    ftypes = {m.space.v.dtype for m in meshes.values()}
    itypes = {m.top.v.dtype for m in meshes.values()}
    if len(ftypes) != 1:
        msg = f"Meshes have different floating point types: {ftypes}"
        return Err(ValueError(msg))
    if len(itypes) != 1:
        msg = f"Meshes have different integer types: {itypes}"
        return Err(ValueError(msg))
    top_elems = {m.top.type for m in meshes.values()}
    if len(top_elems) != 1:
        msg = f"Meshes have different topological element types: {top_elems}"
        return Err(ValueError(msg))
    dims = {m.space.v.shape[1] for m in meshes.values()}
    return Ok(
        _MeshInformation(
            ftypes.pop(),
            itypes.pop(),
            dims.pop(),
            top_elems.pop(),
        )
    )


class _MeshPermutations[I: np.integer](NamedTuple):
    nnodes: int
    nelems: int
    node: Mapping[int, IndexPermutation[I]]
    elem: Mapping[int, IndexPermutation[I]]


def _get_permutations[F: np.floating, I: np.integer](
    meshes: Mapping[int, CheartMesh[F, I]],
) -> Result[_MeshPermutations[I]]:
    nnodes = 0
    nelems = 0
    space_sizes = {k: (nnodes, nnodes := nnodes + int(m.space.n)) for k, m in meshes.items()}
    top_sizes = {k: (nelems, nelems := nelems + int(m.top.n)) for k, m in meshes.items()}
    dtype = {m.top.v.dtype for m in meshes.values()}.pop()
    node_perms = {
        k: create_index_permutation(np.arange(meshes[k].space.n, dtype=dtype), first=i)
        for k, (i, _) in space_sizes.items()
    }
    elem_perms = {
        k: create_index_permutation(np.arange(meshes[k].top.n, dtype=dtype), first=i)
        for k, (i, _) in top_sizes.items()
    }
    return Ok(_MeshPermutations(nnodes, nelems, node_perms, elem_perms))


def _create_interface_mesh[F: np.floating, I: np.integer](
    perms: _MeshPermutations[I], ftype: DType[F], dtype: DType[I]
) -> Result[CheartMesh[F, I]]:
    mask = np.full((perms.nelems, 1), -1, dtype)
    for k, p in perms.elem.items():
        mask[p.new] = k
    if np.any(mask < 0):
        msg = "Logic error: element missing from mask. Check with developer"
        return Err(ValueError(msg))
    interface_x = CheartMeshSpace(v=np.array(list(perms.elem.keys()), dtype=ftype))
    interface_t = CheartMeshTopology(v=mask, type=CheartEnum.VERTEX)
    return Ok(CheartMesh(space=interface_x, top=interface_t, bnd=CheartMeshBoundary[I]({})))


def _create_new_mesh[F: np.floating, I: np.integer](
    meshes: Mapping[int, CheartMesh[F, I]],
    perms: _MeshPermutations[I],
    data: _MeshInformation[F, I],
) -> CheartMesh[F, I]:
    new_x = np.zeros((perms.nnodes, data.dim), dtype=data.ftype)
    for k, p in perms.node.items():
        new_x[p.new] = meshes[k].space.v[p.old]
    new_t = np.zeros((perms.nelems, get_element_size(data.elem)), dtype=data.itype)
    for k, p in perms.elem.items():
        new_t[p.new] = meshes[k].top.v[p.old]
    new_patches = {
        tag: CheartMeshPatch(
            tag=tag, k=perms.elem[k].fwd[b.k], v=perms.node[k].fwd[b.v], type=b.type
        )
        for k, m in meshes.items()
        for tag, b in m.bnd.v.items()
    }
    return CheartMesh(
        space=CheartMeshSpace(v=new_x),
        top=CheartMeshTopology(v=new_t, type=data.elem),
        bnd=CheartMeshBoundary(v=new_patches),
    )


def merge_cheart_meshes[F: np.floating, I: np.integer](
    meshes: Mapping[int, CheartMesh[F, I]] | Sequence[CheartMesh[F, I]],
) -> Result[MergedMesh[F, I]]:
    """Return a dataclass containing the merged mesh.

    Parameters
    ----------
    meshes : Mapping[T, CheartMesh[F, I]]
        A mapping of mesh names to CheartMesh objects to be merged.

    Returns
    -------
    Result[MergedMesh[T, F, I]]
        A Result object containing the MergedMesh dataclass with the merged mesh, interface mesh,
        and index permutations for each original mesh.

    """
    meshes = meshes if isinstance(meshes, Mapping) else dict(enumerate(meshes))
    match _check_mesh_information(meshes):
        case Ok(data): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match _get_permutations(meshes):
        case Ok(perms): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip

    match _create_interface_mesh(perms, data.ftype, data.itype):
        case Ok(iface): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    new_mesh = _create_new_mesh(meshes, perms, data)
    return Ok(MergedMesh(mesh=new_mesh, iface=iface, node_perm=perms.node, elem_perm=perms.elem))


def _is_str_dict[T](d: Mapping[int, T] | Mapping[str, T]) -> TypeIs[Mapping[str, T]]:
    return all(isinstance(k, str) for k in d)


def _organize_variables[V: np.floating](
    variables: Mapping[int, Mapping[str, A2[V]]] | Mapping[str, Mapping[int, A2[V]]],
) -> Result[Mapping[str, Mapping[int, A2[V]]]]:
    if not variables:
        return Err(ValueError("No variables provided"))
    if _is_str_dict(variables):
        return Ok(variables)
    return Ok(
        {v: {k: items[v] for k, items in variables.items() if v in items} for v in variables[0]}
    )


def _check_variable_shape_type[V: np.floating](
    variables: Mapping[str, Mapping[int, A2[V]]],
) -> Result[Mapping[str, tuple[int, np.dtype]]]:
    dtypes = {k: {v.dtype for v in items.values()} for k, items in variables.items()}
    shapes = {k: {v.shape[1] for v in items.values()} for k, items in variables.items()}
    for v, d in dtypes.items():
        if len(d) != 1:
            msg = f"Variable {v} has different dtypes: {d}"
            return Err(ValueError(msg))
    for v, s in shapes.items():
        if len(s) != 1:
            msg = f"Variable {v} has different shapes: {s}"
            return Err(ValueError(msg))
    return Ok({k: (shapes[k].pop(), dtypes[k].pop()) for k in variables})


def merge_point_variables[F: np.floating, I: np.integer, V: np.floating](
    mesh: MergedMesh[F, I],
    variables: Mapping[int, Mapping[str, A2[V]]] | Mapping[str, Mapping[int, A2[V]]],
) -> Result[Mapping[str, A2[V]]]:
    """Merge variable fields on points from multiple meshes.

    Parameters
    ----------
    mesh : MergedMesh[F, I]
        The merged mesh containing the index permutations for each original mesh.
    variables : Mapping[int, Mapping[str, Arr[SAny, V]]]
        A mapping of mesh names to variable fields, where each variable field is a mapping of
        variable names to arrays.

    Returns
    -------
    Result[Mapping[str, Arr[SAny, V]]]
        A Result object containing the merged variable fields.

    """
    match _organize_variables(variables):
        case Ok(_vars): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match _check_variable_shape_type(_vars):
        case Ok(vs): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    merged_vars = {v: np.zeros((mesh.mesh.space.n, n), dtype=t) for v, (n, t) in vs.items()}
    for v, items in _vars.items():
        for k, val in items.items():
            perm = mesh.node_perm[k]
            merged_vars[v][perm.new] = val[perm.old]
    return Ok(merged_vars)


def merge_cell_variables[F: np.floating, I: np.integer, V: np.floating](
    mesh: MergedMesh[F, I],
    variables: Mapping[int, Mapping[str, A2[V]]] | Mapping[str, Mapping[int, A2[V]]],
) -> Result[Mapping[str, A2[V]]]:
    """Merge variable fields on cells from multiple meshes.

    Parameters
    ----------
    mesh : MergedMesh[F, I]
        The merged mesh containing the index permutations for each original mesh.
    variables : Mapping[int, Mapping[str, Arr[SAny, V]]]
        A mapping of mesh names to variable fields, where each variable field is a mapping of
        variable names to arrays.

    Returns
    -------
    Result[Mapping[str, Arr[SAny, V]]]
        A Result object containing the merged variable fields.

    """
    match _organize_variables(variables):
        case Ok(_vars): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match _check_variable_shape_type(_vars):
        case Ok(vs): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    merged_vars = {v: np.zeros((mesh.mesh.top.n, n), dtype=t) for v, (n, t) in vs.items()}
    for v, items in _vars.items():
        for k, val in items.items():
            perm = mesh.elem_perm[k]
            merged_vars[v][perm.new] = val[perm.old]
    return Ok(merged_vars)
