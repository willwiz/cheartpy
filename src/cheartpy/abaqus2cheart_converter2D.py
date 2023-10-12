#!/usr/bin/env python3

import enum
import os
import argparse
from argparse import RawTextHelpFormatter
from typing import Any, Optional, TextIO, Tuple, TypeVar, Union, Self
import numpy as np
from numpy import zeros, ndarray as Arr, array
import dataclasses as dc
from concurrent import futures

i32 = np.dtype[np.int32]
f64 = np.dtype[np.float64]

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
    "-o",
    "--output-file",
    type=str,
    default=None,
    dest="o",
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
    "-c",
    "--cores",
    default=1,
    type=int,
    help="""Enable multiprocessing with n cores
    """,
)


def dimensions(a) -> list[int]:
    if type(a) == list:
        return [len(a)] + dimensions(a[0])
    elif type(a) == dict:
        return [len(a)] + dimensions(next(iter(a.values())))
    else:
        return []


@dc.dataclass(slots=True)
class BoundaryItem:
    name: str
    id: str


@dc.dataclass(slots=True)
class BoundaryPatch:
    elem: int
    nodes: list[int]
    id: str

    def values(self) -> Arr[i32]:
        return [self.elem, *self.nodes, int(self.id)]

    def __hash__(self) -> int:
        return hash((self.id, self.elem, *self.nodes))

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, BoundaryPatch):
            return (
                self.id == __value.id
                and self.elem == __value.elem
                and self.nodes == __value.nodes
            )
        return False

    def __ne__(self, __value: object) -> bool:
        if isinstance(__value, BoundaryPatch):
            if self.id != __value.id:
                return False
            elif self.elem != __value.elem:
                return False
            elif self.nodes != __value.nodes:
                return False
            return True
        return False


@dc.dataclass(slots=True)
class InputArgs:
    inputs: list[str]
    output: str
    dim: int
    topology: list[str] | None
    boundary: dict[str, list[str]] | None
    cores: int


class AbaqusElementType(enum.Enum):
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
        case [_, 8]:
            return AbaqusElementType.TetQuad3D
        case [_, 4]:
            return AbaqusElementType.Tet3D
        case _:
            raise ValueError(f"type = {type} with dim = {dim} not implemented")


@dc.dataclass(slots=True)
class meshtype_element:
    name: str = "none"
    kind: str = "none"
    n: int = 0
    data: dict[int, list[int]] = dc.field(default_factory=dict)

    def read(self, line: str):
        self.n = self.n + 1
        row = [int(i) for i in line.strip().split(",")]
        self.data[row[0]] = row[1:]


@dc.dataclass(slots=True, order=True)
class meshtype_node:
    n: int = 0
    data: dict[int, list[float]] = dc.field(default_factory=dict)

    def read(self, line: str):
        self.n = self.n + 1
        row = line.strip().split(",")
        self.data[int(row[0])] = [float(i) for i in row[1:]]


@dc.dataclass(slots=True)
class meshtype_space:
    n: int
    data: Arr[tuple[int, int], f64]


@dc.dataclass(slots=True)
class meshtype_topology:
    n: int
    data: Arr[tuple[int, int], i32]


@dc.dataclass(slots=True)
class meshtype_boundary_patch:
    n: int
    data: list[list[i32]]

    def __add__(self, other: Self):
        if isinstance(other, self.__class__):
            n = self.n + other.n
            data = self.data + other.data
            return meshtype_boundary_patch(n=n, data=data)
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: {self.__class__} and {type(other)}"
            )


@dc.dataclass(slots=True)
class meshtype_boundary:
    n: int
    data: Arr[tuple[int, int], i32]


class meshtype_print:
    def read(self, line: str):
        print(line)


def make_topology(
    elem: dict[str, meshtype_element], elmap: dict[int, int], tops: list[str]
) -> meshtype_topology:
    n = 0
    data = list()
    for top in (elem[t] for t in tops):
        arraydim = dimensions(top.data)
        el = get_abaqus_element(top.kind, arraydim[1])
        for i in top.data.values():
            n = n + 1
            vals = [v for v in i]
            data.append([elmap[vals[j]] for j in el.value])
    return meshtype_topology(n, np.ascontiguousarray(data, dtype=int))


def make_boundary_patch(
    elem: meshtype_element,
    elmap: dict[int, int],
    topology: meshtype_topology,
    label: int,
) -> meshtype_boundary_patch:
    arraydim = dimensions(elem.data)
    el = get_abaqus_element(elem.kind, arraydim[1])
    n = 0
    data = list()
    for row in elem.data.values():
        check = False
        patch = [elmap[row[j]] for j in el.value]
        for j in range(topology.n):
            if set(patch).issubset(topology.data[j]):
                if check:
                    print(f">>>WARN: multiple elements found for the patch {row}")
                    print(f">>>WARN: If internal boundary then ignore")
                else:
                    check = True
                data.append([j + 1, *patch, label])
                n = n + 1
        if not check:
            raise ValueError(f">>>ERROR: Element not found for the patch {row}")
    return meshtype_boundary_patch(n=n, data=data)


def serial_execute_make_boundary(
    elems: dict[str, meshtype_element],
    elmap: dict[int, int],
    topology: meshtype_topology,
    boundary: dict[str, list[str]],
) -> list[meshtype_boundary_patch]:
    print(f"Making boundary patch in Serial")
    res = list()
    for k, patches in boundary.items():
        for p in patches:
            res.append(make_boundary_patch(elems[p], elmap, topology, int(k)))
    return res


def parallel_execute_make_boundary(
    elems: dict[str, meshtype_element],
    elmap: dict[int, int],
    topology: meshtype_topology,
    boundary: dict[str, list[str]],
    cores: int,
) -> list[meshtype_boundary_patch]:
    print(f"Making boundary patch in Parallel")
    res = list()
    with futures.ProcessPoolExecutor(cores) as exec:
        future_jobs = {}
        for k, patches in boundary.items():
            for p in patches:
                future_jobs[
                    exec.submit(make_boundary_patch, elems[p], elmap, topology, int(k))
                ] = p
        for future in futures.as_completed(future_jobs):
            string = future_jobs[future]
            try:
                print(f"<<<Boundary created from: {string}")
                res.append(future.result())
            except Exception as e:
                print(e)
                raise
    return res


@dc.dataclass(order=True, slots=True)
class Cmesh(object):
    elmap: dict[int, int]
    dim: int
    space: meshtype_space | None = None
    topology: meshtype_topology | None = None
    boundary: dict[str, meshtype_boundary] | None = None

    def import_space(self, space: meshtype_node) -> None:
        arraydim = len(space.data)
        if not (arraydim == space.n):
            raise ValueError(
                f">>>The dimensions of the data, {arraydim}, does not match {space.n}."
            )
        self.space = meshtype_space(0, zeros((len(self.elmap), self.dim), dtype=float))
        for k, v in self.elmap.items():
            self.space.n = self.space.n + 1
            for j in range(self.dim):
                self.space.data[v - 1, j] = float(space.data[k][j])

    def import_topology(
        self, topology: list[str], elems: dict[str, meshtype_element]
    ) -> None:
        self.topology = make_topology(elems, self.elmap, topology)


def abaqus_importer(f: TextIO) -> tuple[meshtype_node, dict[str, meshtype_element]]:
    space = meshtype_node()
    elems: dict[str, meshtype_element] = dict()
    reader = space
    for line in f:
        match line.lower():
            case s if s.startswith("*heading"):
                reader = meshtype_print()
            case s if s.startswith("*node"):
                print(f"Creating space")
                reader = meshtype_node()
                space = reader
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
                        f">>>ERROR: Elset {setname} or Type {settype} is not define for element set"
                    )
                print(f"Creating element {setname} with type {settype}")
                reader = meshtype_element(name=setname, kind=settype)
                elems[reader.name] = reader
            case s if s.startswith("***"):
                reader = meshtype_print()
            case _:
                reader.read(line)
    return space, elems


def read_abaqus_mesh_to_raw_elems(
    files: list[str],
) -> tuple[meshtype_node, dict[str, meshtype_element]]:
    space = None
    elems: dict[str, meshtype_element] = dict()
    for it in files:
        with open(it, "r") as f:
            sp, el = abaqus_importer(f)
        if space is None:
            space = sp
        else:
            if space != sp:
                raise ImportError("Mesh Nodes do not match")
        elems.update(el)
    if space is None:
        raise ValueError("Node data not found")
    return space, elems


def build_elmap(
    space: meshtype_node,
    elems: dict[str, meshtype_element],
    topology: list[str] | None = None,
) -> dict[int, int]:
    if topology is None:
        dim = len(space.data)
        if not (dim == space.n):
            raise ValueError(
                f">>>The dimensions of the data, {dim}, does not match {space.n}."
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


def check_args(args: argparse.Namespace) -> InputArgs:
    if args.o is None:
        name, _ = os.path.splitext(args.input[0])
    else:
        name: str = args.o
    boundary = split_argslist_to_nameddict(args.boundary)
    return InputArgs(args.input, name, args.dim, args.topology, boundary, args.cores)


def check_element_names_i(
    elems: dict[str, meshtype_element], name: str, kind: str
) -> None:
    if not name in elems:
        raise ValueError(f"{kind} {name} can not be found in abaqus file.")


def check_element_names(
    elems: dict[str, meshtype_element],
    topologies: list[str] | None,
    boundaries: dict[str, list[str]] | None,
) -> None:
    if topologies:
        for name in topologies:
            if not name in elems:
                check_element_names_i(elems, name, "Topology")
    if boundaries:
        for bnd in boundaries.values():
            for name in bnd:
                if not name in elems:
                    check_element_names_i(elems, name, "Boundary")


def main(args=None) -> None:
    args = parser.parse_args(args=args)
    inp = check_args(args)
    print(inp)
    space, elems = read_abaqus_mesh_to_raw_elems(inp.inputs)
    check_element_names(elems, inp.topology, inp.boundary)
    if inp.topology is None:
        elmap = build_elmap(space, elems)
    else:
        elmap = build_elmap(space, elems, inp.topology)
    g = Cmesh(elmap, inp.dim)
    g.import_space(space)
    if inp.topology is not None:
        g.import_topology(inp.topology, elems)
    print(g.space)
    print(g.topology)
    print(len(elmap))


if __name__ == "__main__":
    main()
