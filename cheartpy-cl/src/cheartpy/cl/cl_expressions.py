from typing import TYPE_CHECKING

import numpy as np
from cheartpy.fe.trait import IVariable

if TYPE_CHECKING:
    from pytools.arrays import A1


def ll_str(v: IVariable, b: tuple[float, float, float] | A1[np.floating] | IVariable) -> str:
    match b:
        case (l, c, r):
            return f"max(min(({v}.1{-l:+.8g})/({c - l:.8g}),({r:.8g}-{v}.1)/({r - c:.8g})), 0)"
        case np.ndarray():
            return (
                f"max(min(({v}.1{-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g}-{v}.1)/"
                f"({b[2] - b[1]:.8g})), 0)"
            )
        case IVariable():
            return f"max(min(({v}.1 - {b}.1)/({b}.2 - {b}.1),({b}.3 - {v}.1)/({b}.3 - {b}.2)), 0)"
