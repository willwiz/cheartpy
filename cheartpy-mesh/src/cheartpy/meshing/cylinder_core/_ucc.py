"""Universal Cylindrical Space (UCS) functions for cylindrical CheartMesh."""

import dataclasses as dc
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from pytools.arrays import A1


class CylinderParameters(TypedDict, total=False):
    r_in: float
    r_out: float
    length: float
    n_r: int
    n_q: int
    n_z: int


@dc.dataclass(slots=True, frozen=True)
class UniversalCylindricalCoordinate[F: np.floating = np.floating]:
    a_r: A1[F]
    a_c: A1[F]
    a_z: A1[F]
    v_r: A1[F]
    v_c: A1[F]
    v_z: A1[F]


def generate_ucc_for_cylindrical_mesh[F: np.floating, I: np.integer](
    param: CylinderParameters,
) -> UniversalCylindricalCoordinate[F]: ...
