#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses as dc
import enum
import sys
from argparse import RawTextHelpFormatter
from collections import defaultdict
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Self, TextIO

import numpy as np

from .cheart_mesh.io import (
    chwrite_d_utf,
    chwrite_iarr_utf,
    chwrite_str_utf,
    chwrite_t_utf,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from arraystubs import Arr2

################################################################################################
# Check if multiprocessing is available


parser = argparse.ArgumentParser(
    description="""
    Convert Abaqus mesh to Cheart. Main() can be editted for convenience, see example at
    the bottom. Example inputs:

    Default: Exports all elements with default name as mesh_ele_FE.T files.
      python3 abaqus2cheart.py mesh.inp

    With Topology defined as the element Volume:
      python3 abaqus2cheart.py mesh.inp -t Volume

    With Boundaries:
      Surface 1 labeled as 1
      Surfaces 2 3 4 labeled as 2
      Topology as Volume1 and Volume2
      python3 abaqus2cheart.py mesh.inp -t Volume1 Volume2 -b Surface1 1 -b Surface2 Surface3 2

    Mesh is check for errors if topology and boundary as indicated. Extra nodes are not included.

""",
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument(
    "input",
    nargs="+",
    type=str,
    help="""Name of the .inp file containing the Abaqus mesh. If given after the
    optional arguments -t or -b, -- should be inserted in between to delineate.
    """,
)
parser.add_argument(
    "-d",
    "--dim",
    type=int,
    default=3,
    help="""dimension of the mesh, default 3""",
)
parser.add_argument(
    "-p",
    "--prefix",
    type=str,
    default=None,
    help="""Give the prefix for the output files.""",
)
parser.add_argument(
    "-t",
    "--topology",
    nargs="+",
    default=None,
    help="""Define which volume will be used as the topology. If multiple are given,
    they are appended. E.g.,
    --topology Volume1
    --topology Volume1 Volume2 Volume3 ...
    """,
)
parser.add_argument(
    "-b",
    "--boundary",
    action="append",
    nargs="+",
    default=None,
    help="""Set a boundary give the name of the element and label or name, appended
    numerals, and label. E.g.,
    --boundary Surf1  label
    --boundary Surf1 Surf2 ... label
    """,
)
parser.add_argument(
    "--add-mask",
    action="append",
    nargs="+",
    default=None,
    help="""Add a mask with an given element""",
)
parser.add_argument(
    "-c",
    "--cores",
    default=1,
    type=int,
    help="""Enable multiprocessing with n cores
    """,
)


@dc.dataclass(slots=True)
class Mask:
    name: str
    value: str
    elems: list[str]


@dc.dataclass(slots=True)
class InputArgs:
    inputs: list[str]
    prefix: str
    dim: int
    topology: list[str] | None
    boundary: dict[str, list[str]] | None
    masks: dict[str, Mask] | None
    cores: int


class _AbaqusElement(NamedTuple):
    tag: str
    nodes: tuple[int, ...]


class AbaqusElement(enum.Enum):
    S3R = _AbaqusElement("S3R", (0, 1, 2))
    T3D2 = _AbaqusElement("T3D2", (0, 1))
    T3D3 = _AbaqusElement("T3D3", (0, 1, 2))
    CPS3 = _AbaqusElement("CPS3", (0, 1, 2))
    CPS4 = _AbaqusElement("CPS4", (0, 1, 3, 2))
    CPS4_3D = _AbaqusElement("CPS4_3D", (0, 1, 3, 2, 4, 7, 8, 5, 6))
    C3D4 = _AbaqusElement("C3D4", (0, 1, 3, 2))
    TetQuad3D = _AbaqusElement("TetQuad3D", (0, 1, 3, 2, 4, 5, 7, 6))
    Tet3D = _AbaqusElement("Tet3D", (0, 1, 2, 3))


def get_abaqus_element(tag: str, dim: int) -> AbaqusElement:
    match tag, dim:
        case "T3D2", 2:
            kind = AbaqusElement.T3D2
        case "T3D3", 3:
            kind = AbaqusElement.T3D3
        case "CPS3", 3:
            kind = AbaqusElement.CPS3
        case "CPS4", 4:
            kind = AbaqusElement.CPS4
        case "CPS4", 9:
            kind = AbaqusElement.CPS4_3D
        case "C3D4", 4:
            kind = AbaqusElement.C3D4
        case "S3R", 3:
            kind = AbaqusElement.S3R
        case _, 8:
            kind = AbaqusElement.TetQuad3D
        case _, 4:
            kind = AbaqusElement.Tet3D
        case _:
            msg: str = f"Element type '{type}' with dimension {dim} is not implemented. "
            raise ValueError(msg)
    return kind


@dc.dataclass(slots=True)
class AbaqusMeshElement:
    name: str = "none"
    kind: str = "none"
    n: int = 0
    data: dict[int, list[int]] = dc.field(default_factory=dict[int, list[int]])

    def read(self, line: str) -> None:
        self.n = self.n + 1
        row = [int(i) for i in line.strip().split(",")]
        self.data[row[0]] = row[1:]

    def to_numpy(self) -> Arr2[np.intc]:
        return np.ascontiguousarray([*self.data.values()], dtype=np.intc)


@dc.dataclass(slots=True, order=True)
class AbaqusMeshNode:
    n: int = 0
    data: dict[int, list[float]] = dc.field(default_factory=dict[int, list[float]])

    def read(self, line: str) -> None:
        self.n = self.n + 1
        row = line.strip().split(",")
        self.data[int(row[0])] = [float(i) for i in row[1:]]

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.data.keys())))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.data == other.data
        return False


class ReaderPrint:
    def read(self, line: str) -> None:
        print(line)


@dc.dataclass(slots=True)
class MeshTypeSpace[T: np.floating]:
    n: int
    data: Arr2[T]


@dc.dataclass(slots=True)
class MeshTypeTopology[T: np.integer]:
    n: int
    data: Arr2[T]


@dc.dataclass(slots=True)
class MeshTypeBoundary[T: np.integer]:
    n: int
    data: Arr2[T]


@dc.dataclass(slots=True)
class BoundaryElem:
    elem: int
    nodes: list[int]
    tag: int

    def values(self) -> list[int]:
        return [self.elem, *self.nodes, int(self.tag)]

    def __hash__(self) -> int:
        return hash((self.tag, self.elem, *self.nodes))

    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, BoundaryElem):
            return self.tag == value.tag and self.elem == value.elem and self.nodes == value.nodes
        return False


@dc.dataclass(slots=True)
class BoundaryPatch:
    n: int = 0
    data: set[BoundaryElem] = dc.field(default_factory=set[BoundaryElem])

    def __add__(self, other: Self) -> BoundaryPatch:
        if isinstance(other, self.__class__):
            n = self.n + other.n
            data = self.data.union(other.data)
            return BoundaryPatch(n=n, data=data)
        msg = f"unsupported operand type(s) for +: {self.__class__} and {type(other)}"
        raise TypeError(msg)

    def to_numpy(self) -> Arr2[np.intc]:
        return np.ascontiguousarray(
            sorted(
                [v.values() for v in self.data],
                key=lambda x: (x[-1], x[0], x[1:-1]),
            ),
            dtype=np.intc,
        )


@dc.dataclass(order=True, slots=True)
class CheartMesh[T: np.floating, I: np.integer]:
    elmap: dict[int, int]
    dim: int
    space: MeshTypeSpace[T] | None = None
    topology: MeshTypeTopology[I] | None = None
    boundary: MeshTypeBoundary[I] | None = None


def get_results_from_futures(
    func: Callable[..., Any],
    args: list[Any],
    cores: int = 2,
) -> Sequence[Any]:
    with futures.ProcessPoolExecutor(cores) as executor:
        future_jobs = [executor.submit(func, *a) for a in args]
    return [future.result() for future in futures.as_completed(future_jobs)]


def abaqus_importer(f: TextIO) -> tuple[AbaqusMeshNode, dict[str, AbaqusMeshElement]]:
    nodes = AbaqusMeshNode()
    elems: dict[str, AbaqusMeshElement] = {}
    reader = nodes
    for line in f:
        match line.lower():
            case s if s.startswith("*heading"):
                reader = ReaderPrint()
            case s if s.startswith("*node"):
                print("Creating space")
                reader = AbaqusMeshNode()
                nodes = reader
            case s if s.startswith("*element"):
                setheader = line.strip().split(",")
                settype = None
                setname = None
                for h in setheader[1:]:
                    i1, i2 = h.split("=")
                    if i1.lower().strip() == "elset":
                        setname = i2
                    elif i1.lower().strip() == "type":
                        settype = i2
                if setname is None or settype is None:
                    msg = (
                        f">>>ERROR: Elset {setname} or Type {settype} is not define for element set"
                    )
                    raise ImportError(msg)
                print(f"Creating element {setname} with type {settype}")
                reader = AbaqusMeshElement(name=setname, kind=settype)
                elems[reader.name] = reader
            case s if s.startswith("***"):
                reader = ReaderPrint()
            case _:
                reader.read(line)
    return nodes, elems


def read_abaqus_mesh_to_raw_elems(
    files: list[str],
) -> tuple[AbaqusMeshNode, dict[str, AbaqusMeshElement]]:
    nodes = None
    elems: dict[str, AbaqusMeshElement] = {}
    for it in files:
        with Path(it).open("r") as f:
            sp, el = abaqus_importer(f)
        if nodes is None:
            nodes = sp
        elif nodes != sp:
            msg = ">>>ERROR: Mesh Nodes do not match"
            raise ImportError(msg)
        elems.update(el)
    if nodes is None:
        msg = ">>>ERROR: No nodes found in the mesh files"
        raise ValueError(msg)
    return nodes, elems


def build_elmap(
    space: AbaqusMeshNode,
    elems: dict[str, AbaqusMeshElement],
    topology: list[str] | None = None,
) -> dict[int, int]:
    if topology is None:
        dim = len(space.data)
        if dim != space.n:
            msg = f">>>The dimensions of the data, {dim}, does not match {space.n}."
            raise ValueError(msg)
        uniques = set(space.data.keys())
    else:
        uniques = {int(v) for name in topology for vals in elems[name].data.values() for v in vals}
    elmap: dict[int, int] = {}
    nn = 0
    for p in uniques:
        nn = nn + 1
        elmap[p] = nn
    return elmap


def import_space(
    elmap: dict[int, int],
    nodes: AbaqusMeshNode,
    dim: int = 3,
) -> MeshTypeSpace[np.float64]:
    arraydim = len(nodes.data)
    if arraydim != nodes.n:
        msg = f">>>The dimensions of the data, {arraydim}, does not match {nodes.n}."
        raise ValueError(msg)
    space = MeshTypeSpace(0, np.zeros((len(elmap), dim), dtype=np.float64))
    for k, v in elmap.items():
        space.n = space.n + 1
        for j in range(dim):
            space.data[v - 1, j] = float(nodes.data[k][j])
    return space


def import_topology(
    elmap: dict[int, int],
    elems: dict[str, AbaqusMeshElement],
    tops: list[str],
) -> MeshTypeTopology[np.intc]:
    n = 0
    data: list[list[int]] = []
    for top in (elems[t] for t in tops):
        arraydim = len(next(iter(top.data.values())))
        el = get_abaqus_element(top.kind, arraydim)
        for values in top.data.values():
            n = n + 1
            data.append([elmap[values[j]] for j in el.value.nodes])
    return MeshTypeTopology(n, np.ascontiguousarray(data, dtype=np.intc))


def topology_hashmap[T: np.integer](topology: MeshTypeTopology[T]) -> dict[int, set[int]]:
    hashmap: dict[int, set[int]] = defaultdict(set)
    for i, row in enumerate(topology.data, 1):
        for k in row:
            hashmap[int(k)].add(i)
    return hashmap


def find_elem_from_hashmap(node_map: dict[int, set[int]], nodes: list[int]) -> int:
    elem_sets = [node_map[i] for i in nodes]
    elements: set[int] = elem_sets[0].intersection(*elem_sets)
    if len(elements) == 0:
        msg = f">>>ERROR: No element was found containing {nodes}"
        raise ValueError(msg)
    if len(elements) > 1:
        print(
            f">>>WARNNING: Multiple element {elements} was found containing {
                nodes
            }. First one taken.",
        )
    return next(iter(elements))


def make_boundary(
    elmap: dict[int, int],
    topmap: dict[int, set[int]],
    elem: AbaqusMeshElement,
    tag: int,
) -> BoundaryPatch:
    arraydim = len(next(iter(elem.data.values())))
    elem_order = get_abaqus_element(elem.kind, arraydim)
    n = 0
    data: set[BoundaryElem] = set()
    for row in elem.data.values():
        patch = [elmap[row[j]] for j in elem_order.value.nodes]
        k = find_elem_from_hashmap(topmap, patch)
        data.add(BoundaryElem(k, patch, tag))
        n = n + 1
    if n != len(data):
        print(
            f">>>WARNING: Duplicate boundary patch from in {elem.name}, ignored",
            file=sys.stderr,
        )
        n = len(data)
    return BoundaryPatch(n=n, data=data)


def checked_make_boundary(
    elmap: dict[int, int],
    topmap: dict[int, set[int]],
    elem: AbaqusMeshElement,
    tag: int,
) -> BoundaryPatch:
    try:
        bnd = make_boundary(elmap, topmap, elem, tag)
    except Exception as e:
        print(elem.name)
        print(e)
        raise
    return bnd


def import_boundaries(
    elmap: dict[int, int],
    topmap: dict[int, set[int]],
    elems: dict[str, AbaqusMeshElement],
    boundary: dict[str, list[str]],
    cores: int = 1,
) -> MeshTypeBoundary[np.intc]:
    if cores <= 1:
        bnds = [
            checked_make_boundary(elmap, topmap, elems[b], int(k))
            for k, v in boundary.items()
            for b in v
        ]
    else:
        args = [(elmap, topmap, elems[b], int(k)) for k, v in boundary.items() for b in v]
        bnds = get_results_from_futures(checked_make_boundary, args, cores=cores)

    boundaries = BoundaryPatch()
    for b in bnds:
        boundaries = boundaries + b
    if boundaries.n != len(boundaries.data):
        print(
            ">>>WARNING: Duplicate boundary patch found when merging patches, ignored",
            file=sys.stderr,
        )
        boundaries.n = len(boundaries.data)
    return MeshTypeBoundary(boundaries.n, boundaries.to_numpy())


def export_cheart_mesh[T: np.floating, V: np.integer](
    inp: InputArgs,
    g: CheartMesh[T, V],
    nodes: AbaqusMeshNode,
    elems: dict[str, AbaqusMeshElement],
) -> None:
    if g.space is not None:
        chwrite_d_utf(inp.prefix + "_FE.X", g.space.data)
    if g.topology is not None and g.space is not None:
        chwrite_t_utf(inp.prefix + "_FE.T", g.topology.data, g.space.n)
    if g.boundary is not None:
        chwrite_iarr_utf(inp.prefix + "_FE.B", g.boundary.data)
    if inp.topology is None and inp.boundary is None:
        for k, v in elems.items():
            chwrite_t_utf(f"{inp.prefix}_{k}_FE.T", v.to_numpy(), nodes.n)


def split_argslist_to_nameddict(
    varlist: list[list[str]] | None,
) -> dict[str, list[str]] | None:
    if varlist is None:
        return None
    var: dict[str, list[str]] = {}

    for items in varlist:
        if not len(items) > 1:
            msg = ">>>ERROR: Boundary or Topology must have at least 2 items, elem and label."
            raise ValueError(msg)
        var[items[-1]] = items[:-1]
    return var


def check_element_names_i(
    elems: dict[str, AbaqusMeshElement],
    name: str,
    kind: str,
) -> None:
    if name not in elems:
        msg = f">>>ERROR: {kind} {name} not found in the mesh file."
        raise ValueError(msg)


def check_element_names(
    elems: dict[str, AbaqusMeshElement],
    topologies: list[str] | None,
    boundaries: dict[str, list[str]] | None,
) -> None:
    if topologies:
        for name in topologies:
            check_element_names_i(elems, name, "Topology")
    if boundaries:
        if not topologies:
            msg = ">>>ERROR: Boundaries cannot be defined without topology."
            raise ValueError(msg)
        for bnd in boundaries.values():
            for name in bnd:
                check_element_names_i(elems, name, "Boundary")


_MASK_ARG_LEN = 3


def gather_masks(
    cmd_arg_masks: list[list[str]] | None,
) -> dict[str, Mask] | None:
    if cmd_arg_masks is None:
        return None
    masks: dict[str, Mask] = {}
    for m in cmd_arg_masks:
        if len(m) < _MASK_ARG_LEN:
            msg = (
                ">>>ERROR: A Mask must have at least 3 args, e.g., name1, name2, ..., value, output"
            )
            raise ValueError(msg)
        masks[m[-1]] = Mask(m[-1], m[-2], m[:-2])
    return masks


def create_mask(
    elmap: dict[int, int],
    elems: dict[str, AbaqusMeshElement],
    mask: Mask,
) -> None:
    data = np.full((len(elmap), 1), "0", dtype="<U12")
    for elem in (elems[s] for s in mask.elems):
        for vals in elem.data.values():
            for v in vals:
                data[elmap[v] - 1] = mask.value
    chwrite_str_utf(mask.name, data)


def check_args(args: argparse.Namespace) -> InputArgs:
    if args.prefix is None:
        name, _ = Path(args.input[0]).stem
    else:
        name: str = args.prefix
    boundary = split_argslist_to_nameddict(args.boundary)
    masks = gather_masks(args.add_mask)
    return InputArgs(
        args.input,
        name,
        args.dim,
        args.topology,
        boundary,
        masks,
        args.cores,
    )


def main(cmd_args: Sequence[str] | None = None) -> None:
    args = parser.parse_args(args=cmd_args)
    inp = check_args(args)
    nodes, elems = read_abaqus_mesh_to_raw_elems(inp.inputs)
    check_element_names(elems, inp.topology, inp.boundary)
    elmap = build_elmap(nodes, elems, inp.topology)
    g = CheartMesh[np.float64, np.intc](elmap, inp.dim)
    g.space = import_space(g.elmap, nodes, inp.dim)
    if inp.topology is not None:
        g.topology = import_topology(g.elmap, elems, inp.topology)
        top_hashmap = topology_hashmap(g.topology)
        if inp.boundary is not None:
            g.boundary = import_boundaries(
                elmap,
                top_hashmap,
                elems,
                inp.boundary,
                inp.cores,
            )
    if inp.masks:
        for m in inp.masks.values():
            create_mask(elmap, elems, m)
    export_cheart_mesh(inp, g, nodes, elems)


if __name__ == "__main__":
    main()
