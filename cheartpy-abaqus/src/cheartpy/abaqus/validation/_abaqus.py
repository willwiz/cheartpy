from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import AbaqusEnum
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cheartpy.abaqus.reader import AbaqusMesh


def get_abaqus_element(tag: str, dim: int) -> Ok[AbaqusEnum] | Err:
    match tag, dim:
        case "T3D2", 2:
            kind = AbaqusEnum.T3D2
        case "T3D3", 3:
            kind = AbaqusEnum.T3D3
        case "CPS3", 3:
            kind = AbaqusEnum.CPS3
        case "CPS4", 4:
            kind = AbaqusEnum.CPS4
        case "C3D4", 4:
            kind = AbaqusEnum.C3D4
        case "S3R", 3:
            kind = AbaqusEnum.S3R
        case "C3D10", 10:
            kind = AbaqusEnum.C3D10
        case "CPEG6", 6:
            kind = AbaqusEnum.CPEG6
        case _:
            msg = f"Element type '{type}' with dimension {dim} is not implemented. "
            return Err(ValueError(msg))
    return Ok(kind)


def check_for_elements[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I],
    topology: Sequence[str],
    boundary: Mapping[int, Sequence[str]] | None = None,
) -> Ok[None] | Err:
    for name in topology:
        if name not in mesh.elements and name not in mesh.elsets:
            msg = f"Topology '{name}' is not defined in the elements."
            return Err(ValueError(msg))
    if boundary is None:
        return Ok(None)
    for name in boundary.values():
        if name not in mesh.elements and name not in mesh.elsets:
            msg = f"Boundary '{name}' is not defined in the elements."
            return Err(ValueError(msg))
    return Ok(None)
