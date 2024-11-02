#!/usr/bin/env python3

import enum
import os
import sys
from collections import defaultdict
import argparse
from argparse import RawTextHelpFormatter
from typing import Any, Callable, TextIO, Self
import numpy as np
import dataclasses as dc
from concurrent import futures
from cheartpy.var_types import Arr, i32, f64
from .cheart_mesh.io import (
    CHWrite_d_utf,
    CHWrite_t_utf,
    CHWrite_iarr_utf,
    CHWrite_Str_utf,
)

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
      python3 abaqus2cheart.py mesh.inp -t Volume1 Volume2 -b Surface1 1 -b Surface2 Surface3 Surface4 2

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


class AbaqusElementType(enum.Enum):
    S3R = (0, 1, 2)
    T3D2 = (0, 1)
    T3D3 = (0, 1, 2)
    CPS3 = (0, 1, 2)
    CPS4 = (0, 1, 3, 2)
    CPS4_3D = (0, 1, 3, 2, 4, 7, 8, 5, 6)
    C3D4 = (0, 1, 3, 2)
    TetQuad3D = (0, 1, 3, 2, 4, 5, 7, 6)
    Tet3D = (0, 1, 2, 3)


def get_abaqus_element(type: str, dim: int) -> AbaqusElementType:
    match [type, dim]:
        case ["T3D2", 2]:
            return AbaqusElementType.T3D2
        case ["T3D3", 3]:
            return AbaqusElementType.T3D3
        case ["CPS3", 3]:
            return AbaqusElementType.CPS3
        case ["CPS4", 4]:
            return AbaqusElementType.CPS4
        case ["CPS4", 9]:
            return AbaqusElementType.CPS4_3D
        case ["C3D4", 4]:
            return AbaqusElementType.C3D4
        case ["S3R", 3]:
            return AbaqusElementType.S3R
        case [_, 8]:
            return AbaqusElementType.TetQuad3D
        case [_, 4]:
            return AbaqusElementType.Tet3D
        case _:
            raise ValueError(f"type = {type} with dim = {dim} not implemented")


@dc.dataclass(slots=True)
class AbaqusMeshElement:
    name: str = "none"
    kind: str = "none"
    n: int = 0
    data: dict[int, list[int]] = dc.field(default_factory=dict)

    def read(self, line: str):
        self.n = self.n + 1
        row = [int(i) for i in line.strip().split(",")]
        self.data[row[0]] = row[1:]

    def to_numpy(self):
        return np.ascontiguousarray([*self.data.values()], dtype=int)


@dc.dataclass(slots=True, order=True)
class AbaqusMeshNode:
    n: int = 0
    data: dict[int, list[float]] = dc.field(default_factory=dict)

    def read(self, line: str):
        self.n = self.n + 1
        row = line.strip().split(",")
        self.data[int(row[0])] = [float(i) for i in row[1:]]

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other) -> bool:
        if isinstance(other, AbaqusMeshNode):
            return self.data != other.data
        return False


class ReaderPrint:
    def read(self, line: str):
        print(line)


@dc.dataclass(slots=True)
class MeshTypeSpace:
    n: int
    data: Arr[tuple[int, int], f64]


@dc.dataclass(slots=True)
class MeshTypeTopology:
    n: int
    data: Arr[tuple[int, int], i32]


@dc.dataclass(slots=True)
class MeshTypeBoundary:
    n: int
    data: Arr[tuple[int, int], i32]


@dc.dataclass(slots=True)
class BoundaryElem:
    elem: int
    nodes: list[int]
    tag: int

    def values(self) -> list[int]:
        return [self.elem, *self.nodes, int(self.tag)]

    def __hash__(self) -> int:
        return hash((self.tag, self.elem, *self.nodes))

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, BoundaryElem):
            return (
                self.tag == __value.tag
                and self.elem == __value.elem
                and self.nodes == __value.nodes
            )
        return False

    def __ne__(self, __value: object) -> bool:
        if isinstance(__value, BoundaryElem):
            if self.tag != __value.tag:
                return False
            elif self.elem != __value.elem:
                return False
            elif self.nodes != __value.nodes:
                return False
            return True
        return False


@dc.dataclass(slots=True)
class BoundaryPatch:
    n: int = 0
    data: set[BoundaryElem] = dc.field(default_factory=set)

    def __add__(self, other: Self):
        if isinstance(other, self.__class__):
            n = self.n + other.n
            data = self.data.union(other.data)
            return BoundaryPatch(
                n=n,
                data=data,
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: {self.__class__} and {
                    type(other)}"
            )

    def to_numpy(self):
        return np.ascontiguousarray(
            sorted(
                [v.values() for v in self.data], key=lambda x: (x[-1], x[0], x[1:-1])
            ),
            dtype=int,
        )


@dc.dataclass(order=True, slots=True)
class CheartMesh(object):
    elmap: dict[int, int]
    dim: int
    space: MeshTypeSpace | None = None
    topology: MeshTypeTopology | None = None
    boundary: MeshTypeBoundary | None = None


def get_results_from_futures(func: Callable, args: list[Any], cores: int = 2):
    future_jobs = []
    with futures.ProcessPoolExecutor(cores) as exec:
        for a in args:
            future_jobs.append(exec.submit(func, *a))
        try:
            res = [future.result() for future in futures.as_completed(future_jobs)]
        except:
            raise
    return res


def abaqus_importer(f: TextIO) -> tuple[AbaqusMeshNode, dict[str, AbaqusMeshElement]]:
    nodes = AbaqusMeshNode()
    elems: dict[str, AbaqusMeshElement] = dict()
    reader = nodes
    for line in f:
        match line.lower():
            case s if s.startswith("*heading"):
                reader = ReaderPrint()
            case s if s.startswith("*node"):
                print(f"Creating space")
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
                    raise ImportError(
                        f">>>ERROR: Elset {setname} or Type {
                            settype} is not define for element set"
                    )
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
    elems: dict[str, AbaqusMeshElement] = dict()
    for it in files:
        with open(it, "r") as f:
            sp, el = abaqus_importer(f)
        if nodes is None:
            nodes = sp
        else:
            if nodes != sp:
                raise ImportError("Mesh Nodes do not match")
        elems.update(el)
    if nodes is None:
        raise ValueError("Node data not found")
    return nodes, elems


def build_elmap(
    space: AbaqusMeshNode,
    elems: dict[str, AbaqusMeshElement],
    topology: list[str] | None = None,
) -> dict[int, int]:
    if topology is None:
        dim = len(space.data)
        if not (dim == space.n):
            raise ValueError(
                f">>>The dimensions of the data, {
                    dim}, does not match {space.n}."
            )
        uniques = set(space.data.keys())
    else:
        uniques = {
            int(v)
            for name in topology
            for vals in elems[name].data.values()
            for v in vals
        }
    elmap: dict[int, int] = dict()
    nn = 0
    for p in uniques:
        nn = nn + 1
        elmap[p] = nn
    return elmap


def import_space(
    elmap: dict[int, int], nodes: AbaqusMeshNode, dim: int = 3
) -> MeshTypeSpace:
    arraydim = len(nodes.data)
    if not (arraydim == nodes.n):
        raise ValueError(
            f">>>The dimensions of the data, {
                arraydim}, does not match {nodes.n}."
        )
    space = MeshTypeSpace(0, np.zeros((len(elmap), dim), dtype=float))
    for k, v in elmap.items():
        space.n = space.n + 1
        for j in range(dim):
            space.data[v - 1, j] = float(nodes.data[k][j])
    return space


def import_topology(
    elmap: dict[int, int], elems: dict[str, AbaqusMeshElement], tops: list[str]
) -> MeshTypeTopology:
    n = 0
    data = list()
    for top in (elems[t] for t in tops):
        arraydim = len(next(iter(top.data.values())))
        el = get_abaqus_element(top.kind, arraydim)
        for i in top.data.values():
            n = n + 1
            vals = [v for v in i]
            data.append([elmap[vals[j]] for j in el.value])
    return MeshTypeTopology(n, np.ascontiguousarray(data, dtype=int))


def topology_hashmap(topology: MeshTypeTopology) -> dict[int, set[int]]:
    hashmap = defaultdict(set)
    for i, row in enumerate(topology.data, 1):
        for k in row:
            hashmap[k].add(i)
    return hashmap


def find_elem_from_hashmap(map: dict[int, set[int]], nodes: list[int]) -> int:
    elem_sets = [map[i] for i in nodes]
    elements = set.intersection(*elem_sets)
    if len(elements) == 0:
        raise ValueError(f">>>ERROR: No element was found containing {nodes}")
    elif len(elements) > 1:
        print(
            f">>>WARNNING: Multiple element {
                elements} was found containing {nodes}. First one taken."
        )
    return list(elements)[0]


def make_boundary(
    elmap: dict[int, int],
    topmap: dict[int, set[int]],
    elem: AbaqusMeshElement,
    tag: int,
) -> BoundaryPatch:
    arraydim = len(next(iter(elem.data.values())))
    elem_order = get_abaqus_element(elem.kind, arraydim)
    n = 0
    data = set()
    for row in elem.data.values():
        patch = [elmap[row[j]] for j in elem_order.value]
        k = find_elem_from_hashmap(topmap, patch)
        data.add(BoundaryElem(k, patch, tag))
        n = n + 1
    if n != len(data):
        print(
            f">>>WARNING: Duplicate boundary patch from in {
                elem.name}, ignored",
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
) -> MeshTypeBoundary:
    if cores < 2:
        bnds = [
            checked_make_boundary(elmap, topmap, elems[b], int(k))
            for k, v in boundary.items()
            for b in v
        ]
    else:
        args = [
            (elmap, topmap, elems[b], int(k)) for k, v in boundary.items() for b in v
        ]
        bnds = get_results_from_futures(checked_make_boundary, args, cores=cores)

    boundaries = BoundaryPatch()
    for b in bnds:
        boundaries = boundaries + b
    if boundaries.n != len(boundaries.data):
        print(
            f">>>WARNING: Duplicate boundary patch found when merging patches, ignored",
            file=sys.stderr,
        )
        boundaries.n = len(boundaries.data)
    return MeshTypeBoundary(boundaries.n, boundaries.to_numpy())


def export_cheart_mesh(
    inp: InputArgs,
    g: CheartMesh,
    nodes: AbaqusMeshNode,
    elems: dict[str, AbaqusMeshElement],
) -> None:
    if g.space is not None:
        CHWrite_d_utf(inp.prefix + "_FE.X", g.space.data)
    if g.topology is not None and g.space is not None:
        CHWrite_t_utf(inp.prefix + "_FE.T", g.topology.data, g.topology.n, g.space.n)
    if g.boundary is not None:
        CHWrite_iarr_utf(inp.prefix + "_FE.B", g.boundary.data)
    if inp.topology is None and inp.boundary is None:
        for k, v in elems.items():
            CHWrite_t_utf(f"{inp.prefix}_{k}_FE.T", v.to_numpy(), v.n, nodes.n)


def split_argslist_to_nameddict(
    varlist: list[list[str]] | None,
) -> dict[str, list[str]] | None:
    if varlist is None:
        return None
    var: dict[str, list[str]] = dict()

    for items in varlist:
        if len(items) < 2:
            raise ValueError("Not enough values given")
        else:
            var[items[-1]] = items[:-1]
    return var


def check_element_names_i(
    elems: dict[str, AbaqusMeshElement], name: str, kind: str
) -> None:
    if not name in elems:
        raise ValueError(f"{kind} {name} can not be found in abaqus file.")


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
            raise ValueError("Boundaries cannot be defined without topology")
        for bnd in boundaries.values():
            for name in bnd:
                check_element_names_i(elems, name, "Boundary")


def gather_masks(
    cmd_arg_masks: list[list[str]] | None,
) -> dict[str, Mask] | None:
    if cmd_arg_masks is None:
        return None
    masks: dict[str, Mask] = dict()
    for m in cmd_arg_masks:
        if len(m) < 3:
            raise ValueError(
                "A Mask at least 3 args, e.g., name1, name2, ..., value, output"
            )
        masks[m[-1]] = Mask(m[-1], m[-2], m[:-2])
    return masks


def create_mask(
    elmap: dict[int, int], elems: dict[str, AbaqusMeshElement], mask: Mask
) -> None:
    data = np.full((len(elmap), 1), "0", dtype="<U12")
    for elem in (elems[s] for s in mask.elems):
        for vals in elem.data.values():
            for v in vals:
                data[elmap[v] - 1] = mask.value
    CHWrite_Str_utf(mask.name, data)


def check_args(args: argparse.Namespace) -> InputArgs:
    if args.prefix is None:
        name, _ = os.path.splitext(args.input[0])
    else:
        name: str = args.prefix
    boundary = split_argslist_to_nameddict(args.boundary)
    masks = gather_masks(args.add_mask)
    return InputArgs(
        args.input, name, args.dim, args.topology, boundary, masks, args.cores
    )


def main(args=None) -> None:
    args = parser.parse_args(args=args)
    inp = check_args(args)
    nodes, elems = read_abaqus_mesh_to_raw_elems(inp.inputs)
    check_element_names(elems, inp.topology, inp.boundary)
    elmap = build_elmap(nodes, elems, inp.topology)
    g = CheartMesh(elmap, inp.dim)
    g.space = import_space(g.elmap, nodes, inp.dim)
    if inp.topology is not None:
        g.topology = import_topology(g.elmap, elems, inp.topology)
        top_hashmap = topology_hashmap(g.topology)
        if inp.boundary is not None:
            g.boundary = import_boundaries(
                elmap, top_hashmap, elems, inp.boundary, inp.cores
            )
    if inp.masks:
        for _, m in inp.masks.items():
            create_mask(elmap, elems, m)
    export_cheart_mesh(inp, g, nodes, elems)


if __name__ == "__main__":
    main()
