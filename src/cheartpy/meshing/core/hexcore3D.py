from typing import TextIO, Final
import numpy as np
from numpy import ndarray as Arr

f64 = np.dtype[np.float64]
i32 = np.dtype[np.int32]

# This is gives the order of the nodes in an elent given a starting indice
vorder: Final[list[list[int]]] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]


class Mesh:
    n: int
    i: Arr[tuple[int, int], i32]
    v: Arr[tuple[int, int], f64]

    def __getitem__(self, key):
        # if key is of invalid type or value, the list values will raise the error
        return self.v[self.i[key]]


class MeshSpace:
    __slots__ = ["n", "i", "v"]
    n: int
    i: Arr[tuple[int, int, int], i32]
    v: Arr[tuple[int, int], f64]

    def __init__(self, xn: int, yn: int, zn: int, dim: int = 3) -> None:
        self.n = (xn + 1) * (yn + 1) * (zn + 1)
        self.i = np.zeros((xn + 1, yn + 1, zn + 1), dtype=int)
        self.v = np.zeros((self.n, dim))

    def write(self, f: TextIO) -> None:
        for v in self.v:
            for x in v:
                f.write(f"{x:>24.16}")
            f.write("\n")


class MeshTopology:
    __slots__ = ["n", "i", "v"]
    n: int
    i: Arr[tuple[int, int, int], i32]
    v: Arr[tuple[int, int], f64]

    def __init__(self, xn: int, yn: int, zn: int, dim: int = 3, order: int = 1) -> None:
        self.n = xn * yn * zn
        self.i = np.zeros((xn, yn, zn), dtype=int)
        self.v = np.zeros((self.n, (order + 1) ** dim), dtype=int)

    def write(self, f: TextIO) -> None:
        for elem in self.v:
            for node in elem:
                f.write(f"{node + 1:>12d}")
            f.write("\n")


class MeshSurface:
    __slots__ = ["n", "tag", "key", "v"]
    n: int
    tag: int
    key: Arr[int, i32]
    v: Arr[tuple[int, int], f64]

    def __init__(self, npatch: int, tag: int, dim: int = 3, order: int = 1) -> None:
        self.n = npatch
        self.v = np.zeros((npatch, (order + 1) ** (dim - 1)), dtype=int)
        self.key = np.zeros((npatch,), dtype=int)
        self.tag = tag

    def write(self, f: TextIO):
        for k, v in zip(self.key, self.v):
            f.write(f"{k + 1:>12d}")
            for x in v:
                f.write(f"{x + 1:>12d}")
            f.write(f"{self.tag:>12d}\n")


class MeshCheart:
    space: MeshSpace
    top: MeshTopology
    surfs: dict[str, MeshSurface]

    def __init__(self, xn: int, yn: int, zn: int, dim: int = 3, order: int = 1) -> None:
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.space = MeshSpace(xn, yn, zn, dim=dim)
        self.top = MeshTopology(xn, yn, zn, dim=dim, order=order)
        self.surfs = dict()

    def write(self, prefix: str):
        with open(prefix + "_FE.X", "w") as f:
            print(f"Writing space to {f.name}")
            f.write(f"{self.space.n:>12d}{3:>12d}\n")
            self.space.write(f=f)

        with open(prefix + "_FE.T", "w") as f:
            print(f"Writing topology to {f.name}")
            f.write(f"{self.top.n:>12d}{self.space.n:>12d}\n")
            self.top.write(f=f)

        with open(prefix + "_FE.B", "w") as f:
            print(f"Writing boundary to {f.name}")
            npatch = sum([v.n for v in self.surfs.values()])
            f.write(f"{npatch:>12d}\n")
            for _, surf in self.surfs.items():
                surf.write(f=f)


def build_elmap(g: MeshCheart) -> tuple[dict[int, int], int]:
    uniques = np.unique(g.top.v)
    elmap: dict[int, int] = dict()
    nn = 0
    for p in uniques:
        elmap[p] = nn
        nn = nn + 1
    return elmap, nn


def renormalized_mesh(g: MeshCheart) -> MeshCheart:
    elmap, nn = build_elmap(g)
    new_space = np.zeros((nn, g.space.v.shape[1]), dtype=float)
    for k, v in elmap.items():
        new_space[v] = g.space.v[k]
    g.space.n = nn
    g.space.v = new_space
    for i, row in enumerate(g.top.v):
        for j, v in enumerate(row):
            g.top.v[i, j] = elmap[v]
    for b in g.surfs.values():
        for i, row in enumerate(b.v):
            for j, v in enumerate(row):
                b.v[i, j] = elmap[v]
    return g


def mid_squish_transform(x: Arr[int, f64]):
    return x * (2.0 + x * (2.0 * x - 3.0))


def linear_transform(x: Arr[int, f64]):
    return x * (2.0 + x * (2.0 * x - 3.0))
