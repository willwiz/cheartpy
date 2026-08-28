from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pytools.arrays import A2


def normalize_by_row[F: np.floating](vals: A2[F]) -> A2[F]:
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    # norm[norm < _DBL_TOL] = 1.0
    return vals / norm[:, np.newaxis]


def orthonormalize_by_row[F: np.floating](vals: A2[F]) -> A2[F]:
    """Orthonormalize a set of vectors by row using the Gram-Schmidt process."""
    # Create an empty array to hold the orthonormalized vectors
    tensor_arr = vals.reshape(-1, 3, 3)
    q, _ = np.linalg.qr(tensor_arr.mT)
    return q.mT.reshape(-1, 9)
