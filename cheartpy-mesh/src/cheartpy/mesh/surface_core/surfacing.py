import dataclasses as dc
from collections import ChainMap
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
from cheartpy.mesh.struct import CheartMesh, CheartMeshBoundary, CheartMeshPatch
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from pytools.arrays import A1, A2

_COMPONENT = Literal["X", "Y", "Z"]
_COMPONENT_2_INDEX = {"X": 0, "Y": 1, "Z": 2}
_BOUND = (
    tuple[float, float]
    | tuple[_COMPONENT, float]
    | tuple[float, _COMPONENT]
    | tuple[_COMPONENT, _COMPONENT]
)
_SURF_CONSTRAINTS = Mapping[_COMPONENT, _BOUND]


@dc.dataclass(slots=True, frozen=True)
class _NodeMap[F: np.floating, I: np.integer]:
    idx: A1[I]
    x: A2[F]


def _find_nodes_in_constraints_float_float[F: np.floating, I: np.integer](
    node_map: _NodeMap[F, I], component: _COMPONENT, bounds: tuple[float, float]
) -> Ok[_NodeMap[F, I]] | Err:
    lower, upper = bounds
    vals = node_map.x[:, _COMPONENT_2_INDEX[component]]
    valid = (vals >= lower) & (vals <= upper)
    return Ok(_NodeMap(node_map.idx[valid], node_map.x[valid]))


def _find_nodes_in_constraints_var_float[F: np.floating, I: np.integer](
    node_map: _NodeMap[F, I], component: _COMPONENT, bounds: tuple[_COMPONENT, float]
) -> Ok[_NodeMap[F, I]] | Err:
    lower, upper = bounds
    if component == lower:
        msg = "Lower bound cannot be the same as the component."
        return Err(ValueError(msg))
    vals = node_map.x[:, _COMPONENT_2_INDEX[component]]
    lower = node_map.x[:, _COMPONENT_2_INDEX[lower]]
    valid = (vals >= lower) & (vals <= upper)
    return Ok(_NodeMap(node_map.idx[valid], node_map.x[valid]))


def _find_nodes_in_constraints_float_var[F: np.floating, I: np.integer](
    node_map: _NodeMap[F, I], component: _COMPONENT, bounds: tuple[float, _COMPONENT]
) -> Ok[_NodeMap[F, I]] | Err:
    lower, upper = bounds
    if component == upper:
        msg = "Upper bound cannot be the same as the component."
        return Err(ValueError(msg))
    vals = node_map.x[:, _COMPONENT_2_INDEX[component]]
    upper = node_map.x[:, _COMPONENT_2_INDEX[upper]]
    valid = (vals >= lower) & (vals <= upper)
    return Ok(_NodeMap(node_map.idx[valid], node_map.x[valid]))


def _find_nodes_in_constraints_var_var[F: np.floating, I: np.integer](
    node_map: _NodeMap[F, I], component: _COMPONENT, bounds: tuple[_COMPONENT, _COMPONENT]
) -> Ok[_NodeMap[F, I]] | Err:
    lower, upper = bounds
    if component == upper:
        msg = "Upper bound cannot be the same as the component."
        return Err(ValueError(msg))
    if component == lower:
        msg = "Lower bound cannot be the same as the component."
        return Err(ValueError(msg))
    if lower == upper:
        msg = "Lower and upper bounds cannot be the same."
        return Err(ValueError(msg))
    vals = node_map.x[:, _COMPONENT_2_INDEX[component]]
    upper = node_map.x[:, _COMPONENT_2_INDEX[upper]]
    lower = node_map.x[:, _COMPONENT_2_INDEX[lower]]
    valid = (vals >= lower) & (vals <= upper)
    return Ok(_NodeMap(node_map.idx[valid], node_map.x[valid]))


def _find_nodes_in_constraint[F: np.floating, I: np.integer](
    node_map: _NodeMap[F, I], component: _COMPONENT, bounds: _BOUND
) -> Ok[_NodeMap[F, I]] | Err:
    match bounds:
        case float() | int(), float() | int():
            return _find_nodes_in_constraints_float_float(node_map, component, bounds).next()
        case str(), float() | int():
            return _find_nodes_in_constraints_var_float(node_map, component, bounds).next()
        case float() | int(), str():
            return _find_nodes_in_constraints_float_var(node_map, component, bounds).next()
        case str(), str():
            return _find_nodes_in_constraints_var_var(node_map, component, bounds).next()


def find_nodes_in_constraints[F: np.floating, I: np.integer](
    node_map: _NodeMap[F, I], constraints: _SURF_CONSTRAINTS
) -> Ok[_NodeMap[F, I]] | Err:
    current_map = node_map
    for comp, bound in constraints.items():
        match _find_nodes_in_constraint(current_map, comp, bound):
            case Ok(res):
                current_map = res
            case Err(e):
                return Err(e)
    return Ok(current_map)


_SURF_PATCH_INFO = tuple[int, tuple[int, ...]]
_SURF_PATCH_INDEX = Mapping[int, _SURF_PATCH_INFO]


def find_unique_surf_patches[I: np.integer](
    bnd: CheartMeshBoundary[I],
) -> _SURF_PATCH_INDEX:
    return ChainMap(
        *[
            {hash(p := (int(k), tuple(v))): p for (k, v) in zip(b.k, b.v, strict=True)}
            for b in bnd.v.values()
        ]
    )


def find_surface_in_mesh[F: np.floating, I: np.integer](
    bnd: CheartMeshBoundary[I], node_map: _NodeMap[F, I], label: int
) -> CheartMeshPatch[I]:
    nodes_set = set(node_map.idx)
    surf_patches_index = find_unique_surf_patches(bnd)
    dtype = bnd.v[next(iter(bnd.v))].v.dtype
    patchs = {e: b for (e, b) in surf_patches_index.values() if set(b).issubset(nodes_set)}
    k = np.array(list(patchs.keys()), dtype=dtype)
    v = np.array(list(patchs.values()), dtype=dtype)
    return CheartMeshPatch(tag=label, n=len(patchs), k=k, v=v, TYPE=bnd.TYPE)


def create_new_surface_in_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], constraints: _SURF_CONSTRAINTS, label: int
) -> Ok[CheartMesh[F, I]] | Err:
    if mesh.bnd is None:
        msg = "Mesh has no boundary."
        return Err(ValueError(msg))
    if label in mesh.bnd.v:
        msg = f"Boundary with label {label} already exists in the mesh."
        return Err(ValueError(msg))
    nodes = np.unique(mesh.top.v)
    space = mesh.space.v[nodes]
    node_map = _NodeMap(idx=nodes, x=space)
    match find_nodes_in_constraints(node_map, constraints):
        case Ok(node_map):
            new_bnd = find_surface_in_mesh(mesh.bnd, node_map, label)
        case Err(e):
            return Err(e)
    new_mesh = CheartMesh(
        space=mesh.space,
        top=mesh.top,
        bnd=CheartMeshBoundary(
            n=mesh.bnd.n + new_bnd.n,
            v={**mesh.bnd.v, label: new_bnd},
            TYPE=mesh.bnd.TYPE,
        ),
    )
    return Ok(new_mesh)


def create_new_surface_in_surf[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf_id: Iterable[int], constraints: _SURF_CONSTRAINTS, label: int
) -> Ok[CheartMesh[F, I]] | Err:
    if mesh.bnd is None:
        msg = "Mesh has no boundary."
        return Err(ValueError(msg))
    if label in mesh.bnd.v:
        msg = f"Boundary with label {label} already exists in the mesh."
        return Err(ValueError(msg))
    for surf in surf_id:
        if surf not in mesh.bnd.v:
            msg = f"Surface {surf} does not exist in the mesh."
            return Err(ValueError(msg))
    nodes = np.unique([mesh.bnd.v[surf].v for surf in surf_id])
    space = mesh.space.v[nodes]
    node_map = _NodeMap(idx=nodes, x=space)
    match find_nodes_in_constraints(node_map, constraints):
        case Ok(node_map):
            new_bnd = find_surface_in_mesh(mesh.bnd, node_map, label)
        case Err(e):
            return Err(e)
    new_mesh = CheartMesh(
        space=mesh.space,
        top=mesh.top,
        bnd=CheartMeshBoundary(
            n=mesh.bnd.n + new_bnd.n,
            v={**mesh.bnd.v, label: new_bnd},
            TYPE=mesh.bnd.TYPE,
        ),
    )
    return Ok(new_mesh)
