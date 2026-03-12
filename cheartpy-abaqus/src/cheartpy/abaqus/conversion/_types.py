import dataclasses as dc
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import optype.numpy as opn

if TYPE_CHECKING:
    from cheartpy.elem_interfaces import AbaqusEnum
    from pytools.arrays import A1

type ElemSearchMap = Mapping[opn.ToInt, set[int]]
type IndexUpdateMap = Mapping[opn.ToInt, int]


@dc.dataclass(slots=True)
class SpaceIntermediate[F: np.floating]:
    v: Mapping[opn.ToInt, A1[F]]


@dc.dataclass(slots=True)
class ElemIntermediate[I: np.integer]:
    type: AbaqusEnum
    v: MutableMapping[opn.ToInt, A1[I]]


@dc.dataclass(slots=True)
class Mask:
    name: str
    value: str
    elems: Sequence[str]
