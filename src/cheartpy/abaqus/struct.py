import dataclasses as dc
from collections.abc import Mapping, Sequence
from typing import Any, TypeIs

import numpy as np
from arraystubs import Arr2

from .struct import MeshElements
from .trait import AbaqusItem


@dc.dataclass(slots=True)
class Mask:
    name: str
    value: str
    elems: Sequence[str]


@dc.dataclass(slots=True)
class InputArgs:
    inputs: Sequence[str]
    prefix: str
    dim: int
    topology: Sequence[str]
    boundary: Mapping[int, Sequence[str]] | None
    masks: Mapping[str, Mask] | None
    cores: int


@dc.dataclass(slots=True)
class MeshNodes[F: np.floating]:
    n: int
    v: Arr2[F]


def _is_mesh_elements(value: object) -> TypeIs[MeshElements[np.integer[Any]]]:
    return isinstance(value, MeshElements)


@dc.dataclass(slots=True)
class MeshElements[I: np.integer]:
    name: str
    kind: str
    n: int
    v: Arr2[I]

    def __hash__(self) -> int:
        return hash((self.name, self.kind, self.n))

    def __eq__(self, value: object) -> bool:
        if not _is_mesh_elements(value):
            return False
        if self.name != value.name or self.kind != value.kind:
            return False
        if self.n != value.n or not np.array_equal(self.v, value.v):
            return False
        if self.v.dtype != value.v.dtype:
            return False
        if self.v.shape != value.v.shape:
            return False
        return np.array_equal(self.v, value.v)


@dc.dataclass(slots=True)
class AbaqusContent:
    key: AbaqusItem
    value: tuple[str, str] | None = None
