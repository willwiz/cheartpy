from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from cheartpy.cl.mesh import CLPartition


class APIKwargs[F: np.floating](TypedDict, total=False):
    """Keyword arguments for CL FE API functions."""

    partition: CLPartition[F]
    sfx: str
