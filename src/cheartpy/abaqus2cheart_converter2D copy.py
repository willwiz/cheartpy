#!/usr/bin/env python3

import enum
import os
import argparse
from argparse import RawTextHelpFormatter
from typing import Any, Optional, TextIO, Tuple, TypeVar, Union, Self
from numpy import zeros, ndarray, array
from dataclasses import dataclass, field

################################################################################################
# Check if multiprocessing is available
try:
    from concurrent import futures

    futures_avail = True
except:
    futures_avail = False

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
      python3 abaqus2cheart.py mesh.inp -t Volume1 Volume2 -b Surface1 1 -b Surface 2 3 4 2

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
    action="append",
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
    --boundary Surf 1 2 3 4 5 ... label
    """,
)
parser.add_argument(
    "-c",
    "--cores",
    default=None,
    type=int,
    help="""Enable multiprocessing with n cores
    """,
)


def dimensions(a):
    if not type(a) == list:
        return []
    return [len(a)] + dimensions(a[0])


def get_element_info(type: str, dim: int) -> list[int]:
    if type == "T3D2":
        el = [1, 2]
        if not (dim == 3):
            raise ValueError(
                f"dimension of the array imported {dim} doesn't match the given type {type}, should be 3"
            )
    elif type == "T3D3":
        el = [1, 2, 3]
        if not (dim == 4):
            raise ValueError(
                f"dimension of the array imported {dim} doesn't match the given type {type}, should be 4"
            )
    elif type == "CPS3":
        el = [1, 2, 3]
        if not (dim == 4):
            raise ValueError(
                f"dimension of the array imported {dim} doesn't match the given type {type}, should be 4"
            )
    elif type == "CPS4":
        if dim == 5:
            el = [1, 2, 4, 3]
        elif dim == 10:
            el = [1, 2, 4, 3, 5, 8, 9, 6, 7]
    elif type == "C3D4":
        el = [1, 2, 4, 3]
        if not (dim == 5):
            raise ValueError(
                f"dimension of the array imported {dim} doesn't match the given type {type}, should be 5"
            )
    elif dim == 9:
        el = [1, 2, 4, 3, 5, 6, 8, 7]
    elif dim == 5:
        el = [1, 2, 3, 4]
    else:
        raise ValueError(f"Element type {type} not implemented")
    return el


def CHWrite_d_utf(file: str, arr: ndarray) -> None:
    dim = arr.shape
    with open(file, "w") as f:
        f.write("{:12d}".format(dim[0]))
        f.write("{:12d}\n".format(dim[1]))
        for i in arr:
            for j in i:
                f.write("{:>22.12E}".format(j))
            f.write("\n")
    return


def CHWrite_t_utf(file: str, arr: ndarray, ne: int, nn: int) -> None:
    with open(file, "w") as f:
        f.write(f"{ne:12d}")
        f.write(f"{nn:12d}\n")
        for i in arr:
            for j in i:
                f.write(f"{j:>12d}")
            f.write("\n")
    return


def CHWrite_iarr_utf(file: str, arr: ndarray) -> None:
    dim = arr.shape
    with open(file, "w") as f:
        f.write(f"{dim[0]:12d}\n")
        for i in arr:
            for j in i:
                f.write(f"{j:>12d}")
            f.write("\n")
    return


class SearchError(Exception):
    pass


class newdict(dict):
    def __add__(self, other):
        if isinstance(other, self.__class__):
            newd = self.copy()
            newd.update(other)
            return newd
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: {self.__class__} and {type(other)}"
            )


@dataclass(order=True, slots=True)
class mesh(object):
    name: str = field(default="none")
    type: str = field(default="none")
    subc: str = field(default="none")
    n: int = field(default=0)
    data: newdict = field(default_factory=newdict)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            n = self.n + other.n
            data = self.data + other.data
            return type(self)(
                name=self.name, type=self.type, subc=self.subc, n=n, data=data
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: {self.__class__} and {type(other)}"
            )


@dataclass(order=True, slots=True)
class meshtype_element(mesh):
    type: str = field(default="element")

    def read(self, line):
        self.n = self.n + 1
        self.data[self.n] = [int(i) for i in line.strip().split(",")]


@dataclass(order=True, slots=True)
class meshtype_space(mesh):
    type: str = field(default="space")

    def read(self, line):
        self.n = self.n + 1
        row = line.strip().split(",")
        self.data[int(row[0])] = [float(i) for i in row[1:]]


@dataclass(order=True, slots=True)
class meshtype_topology(mesh):
    type: str = field(default="topology")


@dataclass(order=True, slots=True)
class meshtype_boundary(mesh):
    type: str = field(default="boundary")


@dataclass(order=True, slots=True)
class meshtype_print:
    name: str = field(default="none")
    type: str = field(default="none")

    def read(self, line):
        print(line)


@dataclass(order=True, slots=True)
class Cmesh(object):
    dim: int
    nnode: int
    nelem: int
    nbndry: int
    space: ndarray
    topology: ndarray
    boundary: ndarray
    elements: newdict
    elmap: newdict

    def import_space(self, space: meshtype_space, elmap: Optional[dict] = None) -> Self:
        arraydim = len(space.data)
        if not (arraydim == space.n):
            raise ValueError(
                f">>>The dimensions of the data, {arraydim}, does not match {space.n}."
            )
        self.dim = len(next(iter(space.data.values())))
        if elmap is None:
            self.nnode = space.n
            self.elmap = {}
            nn = 0
            for p in space.data.keys():
                nn = nn + 1
                self.elmap[int(p)] = nn
        else:
            self.nnode = len(elmap)
            self.elmap = elmap
        self.space = zeros((self.nnode, self.dim), dtype=float)
        for k, v in self.elmap.items():
            for j in range(self.dim):
                self.space[v - 1, j] = float(space.data[k][j])
        return self

    def import_topology(self, top: meshtype_topology) -> None:
        self.nelem = top.n
        self.topology = array(top.data, dtype=int)
        return

    def import_boundary(self, bndry: meshtype_boundary) -> None:
        self.nbndry = bndry.n
        self.boundary = array(bndry.data, dtype=int)
        return

    def add_element(self, elem: meshtype_element, elmap: Optional[dict] = None) -> None:
        if elmap is None:
            elmap = self.elmap
        ne = elem.n
        arraydim = dimensions(elem.data)
        data = zeros((elem.n, arraydim[1] - 1), dtype=int)
        el = get_element_info(elem.subc, arraydim[1])
        for i in range(elem.n):
            vals = [int(v) for v in elem.data[i]]
            data[i] = [elmap[vals[j]] for j in el[0:]]
        self.elems[elem.name] = meshtype_element(elem.name, elem.subc, ne, data)
        return


def build_elmap(
    space: meshtype_space, topology: Optional[Union[list, meshtype_element]] = None
) -> dict:
    if topology is None:
        dim = len(space.data)
        if not (dim == space.n):
            raise ValueError(
                f">>>The dimensions of the data, {dim}, does not match {space.n}."
            )
        uniques = space.data.keys()
    elif type(topology) is list:
        uniques = set([i for top in topology for j in top.data for i in j[1:]])
    elif type(topology) is meshtype_element:
        uniques = set([i for j in topology.data for i in j[1:]])
    elmap = newdict()
    nn = 0
    for p in uniques:
        nn = nn + 1
        elmap[int(p)] = nn
    return elmap


def make_boundary(
    master: Cmesh,
    elem: meshtype_element,
    k,
    elmap: Optional[newdict] = None,
    name: str = "none",
):
    if elmap is None:
        elmap = master.elmap
    arraydim = dimensions(elem.data)
    el = get_element_info(elem.subc, arraydim[1])
    n = 0
    data = list()
    for row in elem.data:
        check = False
        vals = [int(v) for v in row]
        patch = [elmap[vals[j]] for j in el[0:]]
        for j in range(master.nelem):
            if set(patch).issubset(master.topology[j]):
                if check:
                    print(f">>>WARN: multiple elements found for the patch {row}")
                    print(f">>>WARN: If internal boundary then ignore")
                else:
                    check = True
                data.append([j + 1, *patch, k])
                n = n + 1
        if not check:
            raise SearchError(">>>ERROR: Element not found for the patch {row}")
    return meshtype_boundary(name=name, n=n, data=data)


def make_topology(
    master: Cmesh,
    elem: meshtype_element,
    elmap: Optional[newdict] = None,
    name: str = "none",
) -> None:
    if elmap is None:
        elmap = master.elmap
    arraydim = dimensions(elem.data)
    data = []
    el = get_element_info(elem.subc, arraydim[1])
    n = 0
    for i in elem.data:
        n = n + 1
        vals = [int(v) for v in i]
        data.append([elmap[vals[j]] for j in el[0:]])
    return meshtype_topology(name=name, n=n, data=data)


def importer_sorter(line):
    if line.lower().startswith("*heading"):
        print(f"<<<File being imported:")
        return meshtype_print()
    elif line.lower().startswith("*node"):
        return meshtype_space()
    elif line.lower().startswith("*element"):
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
            print(setname)
            print(settype)
            raise ImportError(">>>ERROR: Elset or Type is not define for element set")
        print(f"Creating element {setname} with type {settype}")
        return meshtype_element(name=setname, subclass=settype)
    elif line.lower().startswith("***"):
        return meshtype_print()
    else:
        raise TypeError(
            f"line does not indicate a structure for {line} should be created. Check code for error."
        )


def abaqus_importer(f: TextIO) -> tuple[meshtype_space, newdict[meshtype_element]]:
    space = None
    elems = newdict()
    reader = None
    for line in f:
        if line.lower().startswith("*"):
            # print(line)
            reader = importer_sorter(line)
            if reader.type == "space":
                space = reader
            elif reader.type == "element":
                elems[reader.name] = reader
        elif reader is not None:
            reader.read(line)
    return space, elems


def main(args=None) -> None:
    args = parser.parse_args(args=args)
    if args.o == None:
        name, _ = os.path.splitext(args.input[0])
    else:
        name = args.o
    space = None
    elems = newdict()
    for it in args.input:
        with open(it, "r") as f:
            sp, el = abaqus_importer(f)
        if space == None:
            space = sp
        else:
            if space != sp:
                raise ImportError("Mesh Nodes do not match")
        elems.update(el)
    if args.topology is None:
        elmap = build_elmap(space=space)
    else:
        try:
            elmap = build_elmap(
                space=space, topology=[elems[str(k)] for k in args.topology]
            )
        except KeyError as e:
            print(
                f"{e} Triggered: {str(k)} cannot be found as an element from the mesh"
            )
        except Exception as e:
            print(e)
            raise
    print(f"<<<Creating mesh from space")
    g = Cmesh().import_space(space=space, elmap=elmap)
    if args.topology is not None:
        top = meshtype_topology(name="top")
        for k in args.topology:
            print(f"<<<Importing topology from {k}")
            try:
                top = top + make_topology(g, elems[str(k)])
            except KeyError as e:
                print(
                    f"{e} Triggered: {str(k)} cannot be found as an element from the mesh"
                )
            except Exception as e:
                print(e)
                raise
        g.import_topology(top)
    if args.boundary is not None:
        print(f"<<<Creating boundaries:")
        bnds = meshtype_boundary(name="bnds")
        if args.cores is None:
            for b in args.boundary:
                if len(b) < 2:
                    raise ValueError(
                        f">>>Each boundary given must be 2 or more strings: [name id] or [name sub1 sub2 ... id"
                    )
                elif len(b) == 2:
                    string = str(b[0])
                    try:
                        bnds = bnds + make_boundary(g, elems[string], str(b[-1]))
                        print(f"<<<Boundary created from: {string}")
                    except KeyError as e:
                        print(
                            f"{e} Triggered: {string} cannot be found as an element from the mesh"
                        )
                    except Exception as e:
                        print(e)
                        raise
                else:
                    for lb in b[1:-1]:
                        string = f"{b[0]}{lb}"
                        try:
                            bnds = bnds + make_boundary(g, elems[string], str(b[-1]))
                            print(f"<<<Boundary created from: {string}")
                        except KeyError as e:
                            print(
                                f"{e} Triggered: {string} cannot be found as an element from the mesh"
                            )
                        except Exception as e:
                            print(e)
                            raise
        elif futures_avail:
            with futures.ProcessPoolExecutor(args.cores) as exec:
                future_jobs = {}
                for b in args.boundary:
                    if len(b) < 2:
                        raise ValueError(
                            f">>>Each boundary given must be 2 or more strings: [name id] or [name sub1 sub2 ... id"
                        )
                    elif len(b) == 2:
                        string = str(b[0])
                        future_jobs[
                            exec.submit(make_boundary, g, elems[string], str(b[-1]))
                        ] = string
                    else:
                        for lb in b[1:-1]:
                            string = f"{b[0]}{lb}"
                            future_jobs[
                                exec.submit(make_boundary, g, elems[string], str(b[-1]))
                            ] = string
                for future in futures.as_completed(future_jobs):
                    string = future_jobs[future]
                    try:
                        print(f"<<<Boundary created from: {string}")
                        bnds = bnds + future.result()
                    except KeyError as e:
                        print(
                            f"{e} Triggered: {string} cannot be found as an element from the mesh"
                        )
                    except Exception as e:
                        print(e)
                        raise
        else:
            raise Exception(
                f"multiprocessing called but concurrent module is not available"
            )
        print(f"<<<Importing created boundary")
        g.import_boundary(bnds)
    CHWrite_d_utf(name + "_FE.X", g.space)
    if args.topology is not None:
        CHWrite_t_utf(name + "_FE.T", g.topology, g.nelem, g.nnode)
    if args.boundary is not None:
        CHWrite_iarr_utf(name + "_FE.B", g.boundary)
    if args.topology is None and args.boundary is None:
        for el in elems.values():
            g.add_element(el)
        for k, v in g.elems.items():
            CHWrite_t_utf(f"{name}_{k}_FE.T", v.data, v.n, g.nnode)


if __name__ == "__main__":
    main()
