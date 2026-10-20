from collections.abc import Mapping

import numpy as np
from cheartpy.elem_interfaces import CheartEnum
from pytools.result import Result

from cheartpy.mesh import CheartMesh

def relabel_cheart_surfaces[
    F: np.floating,
    I: np.integer,
    T: CheartEnum = CheartEnum,
    B: CheartEnum = CheartEnum,
](
    mesh: CheartMesh[F, I, T, B], swap: Mapping[int, int], *, closed: bool = False
) -> Result[CheartMesh[F, I, T, B]]: ...
