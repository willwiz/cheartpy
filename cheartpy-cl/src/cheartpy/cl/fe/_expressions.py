from typing import TYPE_CHECKING

import numpy as np
from cheartpy.fe.trait import IVariable

if TYPE_CHECKING:
    from pytools.arrays import A1


def centerline_shapefunc_expression(
    v: IVariable, b: tuple[float, float, float] | A1[np.floating] | IVariable
) -> str:
    """Return a string expression a triangle element of a for component of cheart defexpression.

    Parameters
    ----------
    v : IVariable
        The variable to be evaluated.
    b : tuple[float, float, float] | A1[np.floating] | IVariable
        The bounds of the triangle element. Options are:
        - A tuple of three floats (l, c, r) representing the left, center, and right bounds.
        - A numpy array of three floats representing the bounds.
        - An IVariable (at least dimension of 3) representing the bounds.

    """
    match b:
        case (l, c, r):
            return f"max(min(({v}{-l:+.8g})/({c - l:.8g}),({r:.8g}-{v})/({r - c:.8g})), 0)"
        case np.ndarray():
            return (
                f"max(min(({v}{-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g}-{v})/"
                f"({b[2] - b[1]:.8g})), 0)"
            )
        case IVariable():
            return f"max(min(({v} - {b}.1)/({b}.2 - {b}.1),({b}.3 - {v})/({b}.3 - {b}.2)), 0)"
