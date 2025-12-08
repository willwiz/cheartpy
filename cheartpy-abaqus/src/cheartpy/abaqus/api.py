from typing import TYPE_CHECKING

from cheartpy.mesh.struct import CheartMesh
from pytools.logging.api import BLogger
from pytools.result import Err, Ok, all_ok

from ._core import (
    build_element_map,
    check_for_elements,
    create_boundaries,
    create_mask,
    create_space,
    create_topology,
    merge_topologies,
    topology_hashmap,
)
from ._readers import read_abaqus_meshes

if TYPE_CHECKING:
    from .struct import InputArgs


def create_cheartmesh_from_abaqus(args: InputArgs) -> None:
    log = BLogger("INFO")
    nodes, elems = read_abaqus_meshes(*args.inputs, log=log)
    check_for_elements(elems, args.topology, args.boundary)
    elmap = build_element_map(elems, args.topology)
    match all_ok([create_topology(elmap, elems, t) for t in args.topology]):
        case Ok(topologies):
            topology = merge_topologies(*topologies)
        case Err(e):
            raise e
    top_hashmap = topology_hashmap(topology)
    if args.boundary:
        match create_boundaries(elmap, top_hashmap, elems, args.boundary):
            case Ok(bnd):
                mesh = CheartMesh(space=create_space(nodes, elmap), top=topology, bnd=bnd)
            case Err(e):
                raise e
    else:
        mesh = CheartMesh(space=create_space(nodes, elmap), top=topology, bnd=None)
    mesh.save(args.prefix)
    if args.masks:
        for m in args.masks.values():
            create_mask(elmap, elems, m)
