from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io import chread_d

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pytools.arrays import A1, DType


def import_region_mask[I: np.integer](file: Path, dtype: DType[I] = np.intp) -> Mapping[int, A1[I]]:
    """Import region mask to dictionary of mask to nodes.

    Parameters
    ----------
    file : Path
        Path to the text file containing the region mask.
    dtype : DType[I], default=np.intp
        The data type of the element IDs.

    Returns
    -------
    Mapping[str, A1[I]]
        A mapping of region names to their corresponding element IDs.

    """
    # Implementation goes here
    mask = chread_d(file, dtype=dtype).flatten()
    return {k: np.nonzero(mask == k)[0].astype(dtype) for k in np.unique(mask)}
