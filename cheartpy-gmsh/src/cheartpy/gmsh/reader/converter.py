import dataclasses as dc
import enum
import itertools
from collections.abc import Mapping, Sequence
from pprint import pprint
from typing import TYPE_CHECKING, Protocol, Unpack

import numpy as np
from cheartpy.elem_interfaces import (
    Cheart2VtkNodeOrder,
    GmshEnum,
    Vtk2Gmsh,
    get_cheart_elem_from_vtk,
)
from cheartpy.gmsh.tools import GmshBoundaryType
from cheartpy.gmsh.types import Entity, GmshMeshTags, Tag
from typing_extensions import TypedDict

import gmsh

from ._types import GmshBndClass, GmshTopInfo, MultiDomainMesh
from .subdomain import split_subdomain

if TYPE_CHECKING:
    from pathlib import Path

    from cheartpy.mesh import CheartMesh, CheartMeshPatch
    from pytools.arrays import A1, A2, DType, ToFloat


class IndexGenerator(Protocol):
    def __call__(self) -> int: ...


class Counter:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> Entity:
        self.count = self.count + 1
        return self.count


new_entity: IndexGenerator = Counter()  # Unique entity tag generator


def add_cheart_master_topology[F: np.floating, I: np.integer](
    mesh: MultiDomainMesh[F, I],
) -> GmshTopInfo:
    n_nodes, dim = mesh.volume.space.v.shape
    node_tags = np.arange(1, n_nodes + 1)
    elem_tags = np.arange(1, mesh.volume.top.n + 1)
    tag = gmsh.model.add_discrete_entity(dim=dim)
    gmsh.model.mesh.add_nodes(
        dim=dim, tag=tag, nodeTags=node_tags, coord=mesh.volume.space.v.flatten()
    )
    vol_elem = Vtk2Gmsh[mesh.volume.top.TYPE]
    cheart_elem = get_cheart_elem_from_vtk(mesh.volume.top.TYPE)
    if cheart_elem is None:
        msg = f"Unsupported element type: {mesh.volume.top.TYPE}"
        raise ValueError(msg)
    element_reorder = Cheart2VtkNodeOrder[cheart_elem]
    print("Reordering element nodes for Gmsh compatibility with : ", element_reorder)
    connectivity = np.ascontiguousarray(mesh.volume.top.v[:, element_reorder] + 1)
    return GmshTopInfo(tag, node_tags, elem_tags, connectivity, vol_elem.value, dim)


def add_boundary_to_gmsh[F: np.floating, I: np.integer](
    top: GmshTopInfo, bnd: CheartMeshPatch[I], current_elem: int = 1
) -> tuple[int, Entity]:
    cheart_elem = get_cheart_elem_from_vtk(bnd.TYPE)
    if cheart_elem is None:
        msg = f"Unsupported boundary element type: {bnd.TYPE}"
        raise ValueError(msg)
    bnd_reorder = Cheart2VtkNodeOrder[cheart_elem]
    bnd_type_id = Vtk2Gmsh[bnd.TYPE].value
    bnd_data = bnd.v[:, bnd_reorder] + 1
    num_bnd_elems = len(bnd_data)
    bnd_tags = np.arange(current_elem, current_elem + num_bnd_elems)
    tag = gmsh.model.add_discrete_entity(dim=top.dim - 1)
    gmsh.model.mesh.add_elements(
        dim=top.dim - 1,
        tag=tag,
        elementTypes=[bnd_type_id],
        elementTags=[bnd_tags],
        nodeTags=[bnd_data.flatten()],
    )
    gmsh.model.add_physical_group(dim=top.dim - 1, tags=[tag], name=f"Surface{bnd.tag}")
    return current_elem + num_bnd_elems, tag


def add_boundaries_to_gmsh[F: np.floating, I: np.integer](
    mesh: MultiDomainMesh[F, I], top: GmshTopInfo, current_elem: int = 1
) -> tuple[int, Mapping[Tag, Entity]]:
    if mesh.volume.bnd is None:
        return current_elem, {}
    bnd_tags = dict[Tag, Entity]()
    for k, v in mesh.volume.bnd.v.items():
        current_elem, tag = add_boundary_to_gmsh(top, v, current_elem)
        bnd_tags[k] = tag
    print("Adding boundaries to model: ", end="")
    pprint(bnd_tags)
    return current_elem, bnd_tags


def add_physical_domain[F: np.floating, I: np.integer](
    top: GmshTopInfo,
    k: int,
    elems: A1[I],
    bnd: Mapping[int, GmshBndClass[I]],
    boundary_map: Mapping[Tag, Entity],
) -> Entity:
    surface_boundaries = [
        boundary_map[k] for k, v in bnd.items() if v["kind"] is GmshBoundaryType.SURF
    ]
    internal_boundaries = [
        boundary_map[k] for k, v in bnd.items() if v["kind"] is GmshBoundaryType.INTERNAL
    ]
    domain_tag = gmsh.model.add_discrete_entity(dim=top.dim, boundary=surface_boundaries)
    gmsh.model.mesh.embed(
        dim=top.dim - 1, tags=internal_boundaries, inDim=top.dim, inTag=domain_tag
    )
    print(f"Attaching boundaries to entity {domain_tag}, name = Elset{k}: ", end="")
    pprint(surface_boundaries)
    pprint(internal_boundaries)
    domain_data = top.connectivity[elems]
    gmsh.model.mesh.add_elements(
        dim=top.dim,
        tag=domain_tag,
        elementTypes=[top.vol_type_id],
        elementTags=[top.elem_tags[elems]],
        nodeTags=[domain_data.flatten()],
    )
    gmsh.model.add_physical_group(dim=top.dim, tags=[domain_tag], name=f"Elset{k}")
    return domain_tag


def add_physical_domains[F: np.floating, I: np.integer](
    mesh: MultiDomainMesh[F, I], top: GmshTopInfo, boundary_map: Mapping[Tag, Entity]
) -> Mapping[Tag, Entity]:
    if mesh.volume.bnd is None:
        msg = "Mesh has no boundary information, cannot add physical domains."
        raise ValueError(msg)
    return {
        k: add_physical_domain(top, k, v, mesh.boundaries[k], boundary_map)
        for k, v in mesh.subdomains.items()
    }


def create_volume(top: GmshTopInfo, domains: Sequence[Entity]) -> Entity:
    """Create a volume from as the combined domains."""
    return gmsh.model.add_physical_group(dim=top.dim, tags=domains, name="Volume1")


class MeshConverterKwargs(TypedDict, total=False):
    optimize: bool
    angle_deg: float


def optimize_mesh(top: GmshTopInfo) -> None:
    """Optimize the mesh using Gmsh's built-in optimization algorithms.

    Parameters
    ----------
    top : GmshTopInfo
        The GmshTopInfo object containing mesh information.

    """
    gmsh.model.occ.synchronize()
    # 5. GENERATE THE MESH
    # gmsh.option.set_number("Mesh.Algorithm", 6)  # 2D Mesh: Frontal-Delaunay
    # gmsh.option.set_number("Mesh.Algorithm3D", 4)  # 3D Mesh: Frontal-Delaunay
    # gmsh.model.mesh.generate(3)  # Use 2 for 2D meshes
    gmsh.option.set_number("Mesh.Optimize", 1)
    gmsh.option.set_number("Mesh.OptimizeNetgen", 1)
    # gmsh.option.set_number("Mesh.OptimizeThreshold", 0.3)  # Target quality bar

    # gmsh.model.mesh.optimize("", force=True, niter=10)
    # gmsh.model.mesh.optimize("UntangleMeshGeometry", force=True, niter=2)
    # match top.dim:
    #     case 3:
    #         gmsh.model.mesh.optimize("Relocate3D", force=True, niter=5)
    #     case 2:
    #         gmsh.model.mesh.optimize("Relocate2D", force=True, niter=5)
    #     case _: ...
    # gmsh.model.mesh.optimize("CGAL")
    gmsh.model.mesh.optimize("Gmsh", force=True, niter=100)
    gmsh.model.mesh.optimize("Netgen", niter=20)


@dc.dataclass(slots=True)
class MeshQualityMetrics:
    rr: A1[np.floating]
    sicn: A1[np.floating]
    inverted: A1[np.floating]


def get_element_mesh_quality(elements: A1[np.integer]) -> MeshQualityMetrics:
    radius_ratios = gmsh.model.mesh.get_element_qualities(elements, qualityName="gamma")
    sicn_values = gmsh.model.mesh.get_element_qualities(elements, qualityName="minSICN")
    inverted_elements = np.array([i for i, v in enumerate(sicn_values) if v < 0])
    return MeshQualityMetrics(np.array(radius_ratios), np.array(sicn_values), inverted_elements)


def get_mesh_quality(
    top: GmshTopInfo, regions: list[int] | None
) -> MeshQualityMetrics | Mapping[int, MeshQualityMetrics]:
    if regions is None:
        return get_element_mesh_quality(top.elem_tags)
    quality_metrics: dict[int, MeshQualityMetrics] = {}
    for elset in regions:
        _, tags, _ = gmsh.model.mesh.get_elements(top.dim, elset)
        quality_metrics[elset] = get_element_mesh_quality(np.array(tags[0]))
    return quality_metrics


class QualityType(enum.StrEnum):
    min = "Minimum"
    max = "Maximum"
    mean = "Mean"
    num = ""


def compute_quality_metric(qual: QualityType, metric: A1[np.floating]) -> ToFloat | int:
    match qual:
        case QualityType.min:
            return np.min(metric)
        case QualityType.mean:
            return np.mean(metric)
        case QualityType.max:
            return np.max(metric)
        case QualityType.num:
            return len(metric)


def print_quality(qual: QualityType, a: A1[np.floating], b: A1[np.floating] | None) -> None:
    u = compute_quality_metric(qual, a)
    v = None if b is None else compute_quality_metric(qual, b)
    match qual:
        case QualityType.num:
            print(f"  (before) {u}", f" (after) {v}" if v else "")
        case _:
            print(f"  {qual:7}", f" (before) {u:8.4f}", f" (after) {v:8.4f}" if v else "")


def print_element_set_quality(before: MeshQualityMetrics, after: MeshQualityMetrics | None) -> None:
    data = dict.fromkeys(
        [QualityType.min, QualityType.mean, QualityType.max],
        (before.rr, after.rr if after else None),
    )
    print("Mesh Quality (Radius Ratio / Gamma):")
    for k, (a, b) in data.items():
        print_quality(k, a, b)
    data = dict.fromkeys(
        [QualityType.min, QualityType.mean, QualityType.max],
        (before.sicn, after.sicn if after else None),
    )
    print("Mesh Quality (Minimum SICN):")
    for k, (a, b) in data.items():
        print_quality(k, a, b)
    print("Inverted Elements:")
    print_quality(QualityType.num, before.inverted, after.inverted if after else None)


type Quality = Mapping[int, MeshQualityMetrics] | MeshQualityMetrics


def print_mesh_quality(before: Quality, after: Quality | None) -> None:
    match before, after:
        case MeshQualityMetrics(), MeshQualityMetrics() | None:
            print_element_set_quality(before, after)
        case Mapping(), Mapping():
            for k in before:
                print(f"Element Set: {k}")
                print_element_set_quality(before[k], after[k] if after else None)
        case Mapping(), MeshQualityMetrics() | None:
            for k, v in before.items():
                print(f"Element Set: {k}")
                print_element_set_quality(v, after)
        case MeshQualityMetrics(), Mapping():
            for k, v in after.items():
                print(f"Element Set: {k}")
                print_element_set_quality(before, v)


def print_physical_groups() -> None:
    # 1. Get all physical groups in the model (returns a list of tuples: (dim, tag))
    physical_groups = gmsh.model.get_physical_groups()

    print(f"{'Dimension':<12} | {'Tag':<5} | {'Physical Group Name'}")
    print("-" * 45)

    # 2. Loop through each group and fetch its metadata
    for dim, tag in physical_groups:
        # Look up the string name assigned to the physical group tag
        name = gmsh.model.get_physical_name(dim, tag)
        # Translate dimension integer to a readable string label
        dim_label = {0: "Node (0D)", 1: "Line (1D)", 2: "Surface (2D)", 3: "Volume (3D)"}.get(
            dim, f"{dim}D"
        )
        print(f"{dim_label:<12} | {tag:<5} | {name}")


def reverse_find_domain_for_boundary[F: np.floating, I: np.integer](
    domain_mesh: MultiDomainMesh[F, I], boundary_tag: int
) -> int:
    for domain_id, bnd in domain_mesh.boundaries.items():
        if boundary_tag in bnd:
            return domain_id
    msg = f"Boundary tag {boundary_tag} not found in any domain."
    raise ValueError(msg)


@dc.dataclass(slots=True)
class GmshElements[I: np.integer]:
    type: GmshEnum
    e: A1[I]
    conn: A2[I]


def get_element[I: np.integer = np.intp](
    el_type: int, tags: Sequence[int], nodes: Sequence[int], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    _, _, _, elem_size, _, _ = gmsh.model.mesh.get_element_properties(el_type)
    return GmshElements(
        type=GmshEnum(el_type),
        e=np.ascontiguousarray(tags, dtype=dtype),
        conn=np.ascontiguousarray(nodes, dtype=dtype).reshape(-1, elem_size),
    )


def merge_elements[I: np.integer](elements: Sequence[GmshElements[I]]) -> GmshElements[I]:
    print(f"Merging {len(elements)} GmshElements...")
    elements = list(elements)

    if not elements:
        msg = "No elements to merge."
        raise ValueError(msg)
    el_types = {el.type for el in elements}
    if len(el_types) != 1:
        msg = f"Cannot merge elements of different types: {el_types}."
        raise ValueError(msg)
    tags = np.concatenate([el.e for el in elements])
    conn = np.concatenate([el.conn for el in elements])
    return GmshElements(type=el_types.pop(), e=tags, conn=conn)


def get_gmsh_entity[I: np.integer = np.intp](
    dim: int, entity_tag: Sequence[Tag], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    print(f"Getting GMSH entity for dimension {dim} and entity tags {entity_tag}...")
    res = list(
        itertools.chain.from_iterable(
            zip(*gmsh.model.mesh.get_elements(dim=dim, tag=i), strict=True) for i in entity_tag
        )
    )
    return merge_elements([get_element(t, tag, nodes, dtype=dtype) for t, tag, nodes in res])


def read_cheartmesh_into_gmsh_api[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    regions: Mapping[int, A1[np.integer]] | None = None,
    **kwargs: Unpack[MeshConverterKwargs],
) -> GmshMeshTags:
    """Convert 3D Volumetric arrays (Tetrahedral or Hexahedral) to Gmsh MSH format.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The 3D volumetric mesh to convert.
    regions : Mapping[str, A1[I]] | None
        Optional mapping of region names to element indices for defining physical groups.
    optimize : bool, default=False
        Whether to optimize the mesh using Gmsh's built-in optimization algorithms.
    angle_deg : float, default=40.0
        The angle threshold in degrees for classifying surfaces during optimization.

    """
    domain_mesh = split_subdomain(mesh, regions).unwrap()
    if not gmsh.is_initialized():
        gmsh.initialize()
    gmsh.model.add("3D_Volumetric_Mesh")

    # 4. ADD PHYSICAL VOLUMES / DOMAINS (Dimension 3)
    top = add_cheart_master_topology(domain_mesh)
    _, boundary_map = add_boundaries_to_gmsh(
        domain_mesh, top, current_elem=int(top.elem_tags.max())
    )
    domains = add_physical_domains(domain_mesh, top, boundary_map)
    print_physical_groups()
    # gmsh.model.mesh.reclassify_nodes()
    gmsh.model.occ.synchronize()
    # gmsh.option.set_number("Mesh.QualityType", 0)
    # Run the plugin safely
    before = get_mesh_quality(top, None)
    print_mesh_quality(before, None)
    old_boundaries = {k: get_gmsh_entity(top.dim - 1, [v]) for k, v in boundary_map.items()}
    old_volume = get_gmsh_entity(top.dim, list(domains.values()))
    if kwargs.get("optimize", False):
        optimize_mesh(top)
        after = get_mesh_quality(top, None)
        gmsh.model.geo.synchronize()
        print_mesh_quality(before, after)
        # if mesh.bnd is not None:
        #     for b in mesh.bnd.v.values():
        #         gmsh.model.add_physical_group(dim=2, tags=[int(b.tag)], tag=100)
    # 3. ADD BOUNDARY SURFACES (Dimension 2)
    volume_tag = create_volume(top, list(domains.values()))
    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.occ.remove_all_duplicates()
    print_physical_groups()
    new_boundaries = {k: get_gmsh_entity(top.dim - 1, [v]) for k, v in boundary_map.items()}
    new_volume = get_gmsh_entity(top.dim, list(domains.values()))
    for k in boundary_map:
        old_bnd = old_boundaries[k]
        new_bnd = new_boundaries[k]
        if not np.array_equal(old_bnd.conn, new_bnd.conn):
            print(f"Boundary {k} has changed after optimization.")
            print(f"Old boundary nodes:\n{old_bnd.conn}")
            print(f"New boundary nodes:\n{new_bnd.conn}")
        else:
            print(f"Boundary {k} remains unchanged after optimization.")

    if not np.array_equal(old_volume.conn, new_volume.conn):
        print("Volume has changed after optimization.")
        print(f"Old volume nodes:\n{old_volume.conn}")
        print(f"New volume nodes:\n{new_volume.conn}")
    else:
        print("Volume remains unchanged after optimization.")
    # gmsh.plugin.run("AnalyseMeshQuality")
    # 5. EXPORT AND FINALIZE
    gmsh.option.set_number("Mesh.SaveGroupsOfNodes", 1)
    print("Successfully imported mesh.")
    boundaries = {
        k: (reverse_find_domain_for_boundary(domain_mesh, k), v) for k, v in boundary_map.items()
    }
    return GmshMeshTags(dim=top.dim, volume=volume_tag, domains=domains, boundarys=boundaries)


def gmsh_finalize(filename: Path | None = None) -> None:
    """Finalize the GMSH API."""
    if not gmsh.is_initialized():
        print("GMSH API is not initialized. No need to finalize.")
    if filename:
        gmsh.write(str(filename))
    else:
        gmsh.fltk.run()
    gmsh.finalize()
