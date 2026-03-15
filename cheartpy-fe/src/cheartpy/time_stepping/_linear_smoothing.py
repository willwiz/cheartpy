from collections.abc import Sequence

import numpy as np
from pytools.arrays import A1, ToFloat


def define_ramp_steps[F: np.floating](
    left: ToFloat, right: ToFloat, dt: ToFloat, *, dtype: DType[F] = np.float64
) -> Sequence[A1[F]]: ...
