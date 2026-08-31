from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from pytools.arrays import A2


def normalize_by_row[F: np.floating](vals: A2[F]) -> Result[A2[F]]:
    """Normalize a set of vectors by row.

    Parameters
    ----------
    vals : A2[F]
        The input array of shape (N, M) where N is the number of vectors and M is the dimension of
        each vector.

    Returns
    -------
    Result[A2[F]]
        A Result object containing the normalized array of shape (N, M) if successful, or
        an error message if any of the vectors are zero and cannot be normalized.

    """
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    if not np.all(norm > 0):
        msg = "Some of all the vectors are zero, cannot normalize."
        return Err(ValueError(msg))
    return Ok(vals / norm[:, np.newaxis])


def orthonormalize_by_row[F: np.floating](vals: A2[F]) -> A2[F]:
    """Orthonormalize a set of vectors by row using the Gram-Schmidt process."""
    # Create an empty array to hold the orthonormalized vectors
    tensor_arr = vals.reshape(-1, 3, 3)
    q, _ = np.linalg.qr(tensor_arr.mT)
    return q.mT.reshape(-1, 9)
