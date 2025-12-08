from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from cheartpy.search.api import get_var_index
from pytools.result import Err, Ok

from .impls import Variable0Getter, Variable1Getter, Variable2Getter
from .traits import IVariableGetter, IVariableList, VarErrors, VarStats

if TYPE_CHECKING:
    from pytools.arrays import A1, A2


__all__ = ["compute_stats", "get_variable_getter"]


def moving_average[T: np.floating](x: A2[T], w: int) -> A1[T]:
    dim = x.shape
    mu = np.zeros(dim[0], dtype=x.dtype)
    for i in range(dim[0]):
        if i < w + 1:
            mu[i] = np.mean(x[: i + w], axis=0)
        elif i > dim[0] - w:
            mu[i] = np.mean(x[i - w - 1 :], axis=0)
        else:
            mu[i] = np.mean(x[i - w - 1 : i + w], axis=0)
    return mu


def compute_stats[T: np.floating](data1: A2[T] | float, data2: A2[T] | float) -> VarStats:
    res = np.ascontiguousarray(data1 - data2)
    rn = max(1, min(len(res) // 10, 10))
    rollmean = moving_average(res, rn)
    rollres = res - rollmean
    rmean = max(rollres.min(), rollres.max(), key=abs)
    norm = np.sqrt(res * res)
    imax = cast("tuple[int, int]", np.unravel_index(res.argmax(), res.shape))
    imin = cast("tuple[int, int]", np.unravel_index(res.argmin(), res.shape))
    vmax = res[imax]
    vmin = res[imin]
    mean_err = norm.mean()
    std = norm.std()
    avg = np.mean(0.5 * (np.abs(data1) + np.abs(data2)))
    return VarStats(float(avg), VarErrors(mean_err, std, vmin, imin, vmax, imax, rmean))


def remove_sfx(name: str) -> str:
    return name[: name.rfind("-")]


def get_idx_for_var(
    var: str | None, root: Path | str | None = None
) -> Ok[IVariableList] | Ok[None] | Err:
    if var is None:
        return Ok(None)
    root = Path(root) if root else Path()
    if var.endswith("*"):
        stem = remove_sfx(var)
        match get_var_index([v.name for v in root.glob(var)], stem):
            case Ok(idx):
                return Ok(IVariableList(stem, idx))
            case Err(e):
                return Err(e)
    return Ok(IVariableList(var, None))


def log_var_error(
    var1: str | None,
    var2: str | None,
) -> None:
    match var1, var2:
        case None, None:
            msg = "No Variable given"
            raise ValueError(msg)
        case None, _:
            msg = f"Variable {var2} not found"
            raise ValueError(msg)
        case _, None:
            msg = f"Variable {var1} not found"
            raise ValueError(msg)
        case _:
            msg = f"Variables {var1} and {var2} not found"
            raise ValueError(msg)


def get_variables(
    var1: str | None,
    var2: str | None,
    root: Path | str | None = None,
) -> Ok[tuple[IVariableList | None, IVariableList | None]] | Err:
    root = Path(root) if root else Path()
    match get_idx_for_var(var1, root):
        case Err(e):
            return Err(e)
        case Ok(v1):
            pass
    match get_idx_for_var(var2, root):
        case Err(e):
            return Err(e)
        case Ok(v2):
            pass
    return Ok((v1, v2))


def _get_getter(name: str, value: list[int] | None, root: Path) -> IVariableGetter:
    match value:
        case None:
            return Variable0Getter(name, None, root)
        case list(v):
            return Variable1Getter(name, None, v, root)


def _get_getter2(
    name1: str,
    name2: str,
    v1: list[int] | None,
    v2: list[int] | None,
    root: Path,
) -> IVariableGetter:
    match v1, v2:
        case None, None:
            return Variable0Getter(name1, name2, root)
        case None, list(v):
            return Variable1Getter(name1, name2, v, root, reverse=True)
        case list(v), None:
            return Variable1Getter(name1, name2, v, root)
        case list(v1), list(v2):
            if set(v1) != set(v2):
                msg = "Indexes do not match"
                raise ValueError(msg)
            return Variable2Getter(name1, name2, v1, root)


def get_variable_getter(
    var1: IVariableList | None,
    var2: IVariableList | None,
    root: Path,
) -> IVariableGetter:
    match var1, var2:
        case None, None:
            msg = f"No Variable given for {var1} and {var2}"
            raise ValueError(msg)
        case (IVariableList(name, v), None) | (None, IVariableList(name, v)):
            return _get_getter(name, v, root)
        case (IVariableList(name=n1, idx=v1), IVariableList(name=n2, idx=v2)):
            return _get_getter2(n1, n2, v1, v2, root)
