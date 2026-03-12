import dataclasses as dc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dc.dataclass(slots=True)
class ParsedInput:
    inputs: Sequence[str]
    prefix: str | None
    dim: int
    topology: Sequence[str]
    boundary: Sequence[str] | None
    add_mask: Sequence[str]
    cores: int


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
