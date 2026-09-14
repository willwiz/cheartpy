from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from pytools.arrays import A2


def check_duplicate_nodes[F: np.floating](space: A2[F]) -> bool:
    """Check for duplicate nodes in the provided array of nodes.

    Parameters
    ----------
    space : A2[F]
        A 2D array of shape (n_nodes, n_dimensions) representing the coordinates of the nodes.

    Returns
    -------
    bool
        True if there are duplicate nodes, False otherwise.

    """
    tree = cKDTree(space)
    diffs = np.diff(np.sort(space, axis=0), axis=0)
    distance = np.linalg.norm(diffs, axis=1).mean()
    pairs = tree.query_pairs(r=1e-6 * distance)  # Adjust the tolerance as needed
    return len(pairs) > 0


def check_duplicate_elements[I: np.integer](connectivity: A2[I]) -> bool:
    """Check for duplicate elements in the provided connectivity array.

    Parameters
    ----------
    connectivity : A2[I]
        A 2D array of shape (n_elements, n_nodes_per_element) representing the connectivity of the
        mesh.

    Returns
    -------
    bool
        True if there are duplicate elements, False otherwise.

    """
    sorted_connectivity = np.sort(connectivity, axis=1)
    void_view = np.ascontiguousarray(sorted_connectivity).view(
        np.dtype((np.void, sorted_connectivity.dtype.itemsize * sorted_connectivity.shape[1]))
    )
    _, unique_indices = np.unique(void_view, return_index=True)
    return len(unique_indices) < len(connectivity)
