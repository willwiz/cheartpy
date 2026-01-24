from typing import TYPE_CHECKING

from cheartpy.mesh.struct import CheartMesh
from pytools.logging import BLogger
from pytools.result import all_ok

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
    from ._struct import InputArgs


def create_cheartmesh_from_abaqus(args: InputArgs) -> None:
    log = BLogger("INFO")
    nodes, elems = read_abaqus_meshes(*args.inputs, log=log).unwrap()
    check_for_elements(elems, args.topology, args.boundary).unwrap()
    elmap = build_element_map(elems, args.topology)
    individual_topologies = all_ok(
        [create_topology(elmap, elems, t) for t in args.topology]
    ).unwrap()
    topology = merge_topologies(*individual_topologies).unwrap()
    top_hashmap = topology_hashmap(topology)
    boundaries = (
        create_boundaries(elmap, top_hashmap, elems, args.boundary).unwrap()
        if args.boundary
        else None
    )
    mesh = CheartMesh(space=create_space(nodes, elmap), top=topology, bnd=boundaries)
    mesh.save(args.prefix)
    if not args.masks:
        return
    for m in args.masks.values():
        create_mask(elmap, elems, m)
