__all__ = ["compute_stats", "get_variable_getter"]
from glob import glob
from ..io.indexing.search import get_var_index
import numpy as np
from typing import cast
from ..var_types import *
from .traits import *
from .impls import *


def moving_average(x: Mat[f64], w: int) -> Vec[f64]:
    dim = x.shape
    mu = np.zeros(dim)
    for i in range(dim[0]):
        if i < w + 1:
            mu[i] = np.mean(x[: i + w], axis=0)
        elif i > dim[0] - w:
            mu[i] = np.mean(x[i - w - 1 :], axis=0)
        else:
            mu[i] = np.mean(x[i - w - 1 : i + w], axis=0)
    return mu


def compute_stats(data1: Mat[f64] | float, data2: Mat[f64] | float) -> VarStats:
    res = np.ascontiguousarray(data1 - data2)
    rn = max(1, min(len(res) // 10, 10))
    rollmean = moving_average(res, rn)
    rollres = res - rollmean
    rmean = max(rollres.min(), rollres.max(), key=abs)
    norm = np.sqrt(res * res)
    imax = cast(tuple[int, int], np.unravel_index(res.argmax(), res.shape))
    imin = cast(tuple[int, int], np.unravel_index(res.argmin(), res.shape))
    vmax = res[imax]
    vmin = res[imin]
    mean_err = norm.mean()
    std = norm.std()
    avg = np.mean(0.5 * (np.abs(data1) + np.abs(data2)))
    return VarStats(float(avg), VarErrors(mean_err, std, vmin, imin, vmax, imax, rmean))


def remove_sfx(name: str) -> str:
    return name[: name.rfind("-")]


def get_idx_for_var(var: str | None, root: str | None = None) -> IVariable | None:
    if var is None:
        return None
    elif var.endswith("*"):
        files = glob(var, root_dir=root)
        return IVariable(remove_sfx(var), get_var_index(files, remove_sfx(var)))
    else:
        return IVariable(var, None)


def get_variable_getter(
    var1: str, var2: str | None, root: str | None = None
) -> IVariableGetter:
    v1 = get_idx_for_var(var1, root)
    v2 = get_idx_for_var(var2, root)
    match v1, v2:
        case None, None:
            raise ValueError("No Variable given")
        case IVariable(idx=v), None:
            if v is None:
                return Variable0Getter(v1.name, v2, root)
            else:
                return Variable1Getter(v1.name, v2, v, root)
        case IVariable(idx=v), IVariable(idx=None):
            if v is None:
                return Variable0Getter(v1.name, v2.name, root)
            else:
                return Variable1Getter(v1.name, v2.name, v, root)
        case None, IVariable(idx=v):
            if v is None:
                return Variable0Getter(v2.name, v1, root)
            else:
                return Variable1Getter(v2.name, v1, v, root, reversed=True)
        case IVariable(idx=None), IVariable(idx=v):
            if v is None:
                return Variable0Getter(v2.name, v1.name, root)
            else:
                return Variable1Getter(v2.name, v1.name, v, root, reversed=True)
        case IVariable(idx=x), IVariable(idx=y):
            if x is None or y is None:
                raise ValueError("Impossible outcome")
            if set(x) != set(y):
                raise ValueError("Indexes do not match")
            return Variable2Getter(v1.name, v2.name, x, root)
