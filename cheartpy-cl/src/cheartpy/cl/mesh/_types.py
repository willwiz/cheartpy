import dataclasses as dc
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from cheartpy.mesh import CheartMesh
    from pytools.arrays import A1, A2


class APIKwargs[F: np.floating](TypedDict, total=False):
    """Keyword arguments for CL API functions."""

    partition: CLPartition[F]


@dc.dataclass(slots=True)
class CLPartition[F: np.floating]:
    prefix: str
    in_surf: int
    node: A1[F]
    domain: A2[F]

    def __repr__(self) -> str:
        return self.prefix

    def __str__(self) -> str:
        return self.prefix


class CLMesh[F: np.floating, I: np.integer](NamedTuple):
    body: CheartMesh[F, I]
    iface: CheartMesh[F, I]
    domain: A2[F]
    elem: A2[F]
