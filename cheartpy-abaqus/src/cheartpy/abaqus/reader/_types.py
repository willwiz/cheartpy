import dataclasses as dc
import enum
from typing import TYPE_CHECKING, Any, Required, TypedDict, TypeIs, get_args, get_origin

import numpy as np

if TYPE_CHECKING:
    from cheartpy.elem_interfaces import AbaqusEnum
    from pytools.arrays import A1, DType, ToInt


class AbaqusHeader(enum.StrEnum):
    HEADINGS = "*heading"
    NODES = "*node"
    ELEMENT = "*element"
    NSET = "*nset"
    ELSET = "*elset"
    COMMENTS = "***"


@dc.dataclass(slots=True)
class Content:
    type: AbaqusHeader
    next: str


def _is_mesh_elements(value: object) -> TypeIs[Element[np.integer[Any]]]:
    return isinstance(value, Element)


@dc.dataclass(slots=True)
class Headings:
    v: list[str]


def _is_node[F: np.floating](value: object, kind: type[F]) -> TypeIs[Nodes[F]]:
    origin = get_origin(value)
    subscript = get_args(value)
    return origin is Nodes and len(subscript) == 1 and isinstance(subscript[0], kind)


@dc.dataclass(slots=True)
class Nodes[F: np.floating]:
    v: dict[ToInt, A1[F]]

    def __hash__(self) -> int:
        return hash(tuple(self.v.keys()))

    def __eq__(self, value: object, /) -> bool:
        dtype = next(iter(self.v.values())).dtype
        if not _is_node(value, dtype.type):
            return False
        if self.v.keys() != value.v.keys():
            return False
        return all(np.array_equal(self.v[k], value.v[k]) for k in self.v)


@dc.dataclass(slots=True)
class Element[I: np.integer]:
    name: str
    type: AbaqusEnum
    v: dict[ToInt, A1[I]]

    def __hash__(self) -> int:
        return hash((self.name, self.type, len(self.v)))

    def __eq__(self, value: object) -> bool:
        if not _is_mesh_elements(value):
            return False
        if self.name != value.name or self.type != value.type:
            return False
        if len(self.v) != len(value.v):
            return False
        if self.v != value.v:
            return False
        return all(np.array_equal(self.v[k], value.v[k]) for k in self.v)


@dc.dataclass(slots=True)
class NSet[I: np.integer]:
    name: str
    v: A1[I]


@dc.dataclass(slots=True)
class ElSet[I: np.integer]:
    name: str
    v: A1[I]


@dc.dataclass(slots=True)
class AbaqusMesh[F: np.floating, I: np.integer]:
    headings: Headings
    nodes: Nodes[F]
    nset: dict[str, NSet[I]]
    elements: dict[str, Element[I]]
    elsets: dict[str, ElSet[I]]
    ftype: DType[F]
    dtype: DType[I]

    def add_headings(self, item: Headings) -> AbaqusMesh[F, I]:
        return AbaqusMesh(
            item, self.nodes, self.nset, self.elements, self.elsets, self.ftype, self.dtype
        )

    def add_nodes(self, item: Nodes[F]) -> AbaqusMesh[F, I]:
        nodes = Nodes(self.nodes.v | item.v)
        return AbaqusMesh(
            self.headings, nodes, self.nset, self.elements, self.elsets, self.ftype, self.dtype
        )

    def add_nset(self, item: NSet[I]) -> AbaqusMesh[F, I]:
        nset = self.nset | {item.name: item}
        return AbaqusMesh(
            self.headings, self.nodes, nset, self.elements, self.elsets, self.ftype, self.dtype
        )

    def add_element(self, item: Element[I]) -> AbaqusMesh[F, I]:
        elements = {item.name: item} | self.elements
        return AbaqusMesh(
            self.headings, self.nodes, self.nset, elements, self.elsets, self.ftype, self.dtype
        )

    def add_elset(self, item: ElSet[I]) -> AbaqusMesh[F, I]:
        elset = {item.name: item} | self.elsets
        return AbaqusMesh(
            self.headings, self.nodes, self.nset, self.elements, elset, self.ftype, self.dtype
        )


class CMDInputKwargs(TypedDict, total=False):
    topology: Required[list[str]]
    prefix: str
    dim: int
    boundary: list[list[str]] | None
    masks: list[list[str]] | None
    cores: int
