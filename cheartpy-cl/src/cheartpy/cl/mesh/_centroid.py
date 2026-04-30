import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cheartpy.mesh import CheartMesh
    from pytools.arrays import A1, A2


@dc.dataclass(slots=True)
class CentroidCurve[F: np.floating]:
    t: A1[F]
    v: A2[F]


def compute_centroid_node[F: np.floating](points: A2[F]) -> A1[F]: ...


def compute_centroid_curve[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], cl: A1[F]
) -> CentroidCurve[F]: ...
