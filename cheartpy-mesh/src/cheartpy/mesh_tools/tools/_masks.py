from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1


def change_mask[I: np.integer](
    mask: A1[I], edit: Mapping[int, int], *, force: bool = False
) -> A1[I]:
    """Change the values of a mask array according to a mapping.

    Parameters
    ----------
    mask : A1[I]
        The input mask array to be modified.

    edit : Mapping[int, int]
        A mapping of old values to new values. Each key-value pair in the mapping
        specifies that all occurrences of the key in the mask should be replaced with
        the corresponding value.

    force : bool, default=False
        If True, the function will raise a ValueError if any of the old values specified
        in the mapping are not found in the mask. Default is False.

    Returns
    -------
    A1[I] | None
        A new mask array with the specified values changed according to the mapping.
        If force is True and any old values are not found in the mask, None is returned.

    """
    if force and not np.isin(edit.keys(), mask).all():
        return None
    new_mask = mask.copy()
    for old_value, new_value in edit.items():
        new_mask[mask == old_value] = new_value
    return new_mask
