from typing import TYPE_CHECKING, Unpack

import numpy as np

if TYPE_CHECKING:
    from cheartpy.cl.mesh import CLDef
    from cheartpy.fe.physics.fs_coupling import FSCouplingProblem

    from ._types import APIKwargs


def create_fs_centerline_problem[F: np.floating](
    defn: CLDef[F], **kwargs: Unpack[APIKwargs[F]]
) -> FSCouplingProblem:
    raise NotImplementedError
