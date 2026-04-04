from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result
from scipy.signal import convolve

from ._traits import VarErrors, VarStats

if TYPE_CHECKING:
    from pytools.arrays import A2


__all__ = ["compute_stats"]


def moving_average[T: np.floating](x: A2[T], w: int) -> A2[T]:
    _, *offaxis = x.shape
    window = np.ones((w * 2 + 1, *offaxis), dtype=x.dtype)
    return convolve(x, window, mode="same") / convolve(np.ones_like(x), window, mode="same")


def compute_stats[T: np.floating](data1: A2[T] | float, data2: A2[T] | float) -> Result[VarStats]:
    if data1.shape != data2.shape:
        return Err(ValueError(f"Data shapes do not match: {data1.shape} vs {data2.shape}"))
    avg = np.mean(0.5 * (np.abs(data1) + np.abs(data2)))
    # residuals
    res = np.ascontiguousarray(data1 - data2)
    norm = np.sqrt(res * res)
    mean_err = norm.mean()
    std = norm.std()
    imax = np.unravel_index(res.argmax(), res.shape)
    imin = np.unravel_index(res.argmin(), res.shape)
    vmax = res[imax]
    vmin = res[imin]
    # check for bias in the rolling mean
    rn = max(1, min(len(res) // 10, 10))
    rollmean = moving_average(res, rn)
    rollres = res - rollmean
    rbias = max(rollres.min(), rollres.max(), key=abs)
    return Ok(VarStats(float(avg), VarErrors(mean_err, std, vmin, imin, vmax, imax, rbias)))


def table_row(it: int | str, res: VarStats) -> str:
    return f"{it:>8}\u2016{res}"
