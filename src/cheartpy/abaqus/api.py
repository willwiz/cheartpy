from pytools.logging.api import BLogger

from cheartpy.abaqus.struct import InputArgs
from cheartpy.cheart_mesh.struct import CheartMesh

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


def create_cheartmesh_from_abaqus(args: InputArgs) -> None:
    log = BLogger("INFO")
    nodes, elems = read_abaqus_meshes(*args.inputs, log=log)
    check_for_elements(elems, args.topology, args.boundary)
    elmap = build_element_map(elems, args.topology)
    topologies = [create_topology(elmap, elems, t) for t in args.topology]
    topology = merge_topologies(*topologies)
    top_hashmap = topology_hashmap(topology)
    mesh = CheartMesh(
        space=create_space(nodes, elmap),
        top=topology,
        bnd=create_boundaries(elmap, top_hashmap, elems, args.boundary) if args.boundary else None,
    )
    mesh.save(args.prefix)
    if args.masks:
        for m in args.masks.values():
            create_mask(elmap, elems, m)
