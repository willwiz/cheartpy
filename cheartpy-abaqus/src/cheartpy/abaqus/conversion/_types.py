import dataclasses as dc
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cheartpy.elem_interfaces import AbaqusEnum
    from pytools.arrays import A1, ToInt

type ElemSearchMap = Mapping[ToInt, set[ToInt]]
type IndexUpdateMap = Mapping[ToInt, ToInt]


@dc.dataclass(slots=True)
class SpaceIntermediate[F: np.floating]:
    v: Mapping[ToInt, A1[F]]


@dc.dataclass(slots=True)
class ElemIntermediate[I: np.integer]:
    type: AbaqusEnum
    v: MutableMapping[ToInt, A1[I]]


@dc.dataclass(slots=True)
class Mask:
    name: str
    value: str
    elems: Sequence[str]
