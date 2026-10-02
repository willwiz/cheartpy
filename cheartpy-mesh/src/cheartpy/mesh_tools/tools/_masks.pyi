from collections.abc import Mapping
from typing import Literal, overload

import numpy as np
from pytools.arrays import A1

@overload
def change_mask[I: np.integer](mask: A1[I], edit: Mapping[int, int]) -> A1[I]: ...
@overload
def change_mask[I: np.integer](
    mask: A1[I], edit: Mapping[int, int], *, force: Literal[True]
) -> A1[I] | None: ...
@overload
def change_mask[I: np.integer](
    mask: A1[I], edit: Mapping[int, int], *, force: Literal[False]
) -> A1[I]: ...
