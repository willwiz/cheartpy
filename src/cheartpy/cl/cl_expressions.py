__all__ = ["LL_str"]
import numpy as np
from ..cheart.trait import IVariable
from ..var_types import *


def LL_str(v: IVariable, b: tuple[float, float, float] | Vec[f64] | IVariable) -> str:
    match b:
        case (l, c, r):
            return (
                f"max(min(({v}{-l:+.8g})/({c - l:.8g}),({r:.8g}-{v})/({r - c:.8g})), 0)"
            )
        case np.ndarray():
            return f"max(min(({v}{-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g}-{v})/({b[2] - b[1]:.8g})), 0)"
        case IVariable():
            return f"max(min(({v} - {b}.1)/({b}.2 - {b}.1),({b}.3 - {v})/({b}.3 - {b}.2)), 0)"
