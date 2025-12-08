from typing import TYPE_CHECKING, Final, ReadOnly, TypedDict, cast

import numpy as np
from cheartpy.io.api import read_array_float
from cheartpy.search.api import get_var_index
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pytools.arrays import A1

__all__ = ["create_time_series_file", "create_time_series_range"]
_CURRENT_VERSION: Final[str] = "1.0.0"


class TimeSeriesItem(TypedDict):
    name: str
    time: float


TIME_SERIES = TypedDict(
    "TIME_SERIES",
    {
        "file-series-version": ReadOnly[str],
        "files": ReadOnly[list[TimeSeriesItem]],
    },
)


def create_json(vtus: Sequence[str], times: A1[np.float64]) -> TIME_SERIES:
    return {
        "file-series-version": _CURRENT_VERSION,
        "files": [{"name": n, "time": t} for n, t in zip(vtus, times, strict=False)],
    }


def create_time_series_core(
    prefix: str,
    vtus: Sequence[str],
    times: A1[np.float64],
) -> Ok[TIME_SERIES] | Err:
    match get_var_index(vtus, prefix, "vtu"):
        case Err(e):
            return Err(e)
        case Ok(idx):
            times = times[idx]
    return Ok(create_json(vtus, times))


def create_time_series_file(prefix: str, time: str, root: Path) -> Ok[TIME_SERIES] | Err:
    vtus = root.glob(f"{prefix}-*.vtu")
    times = read_array_float(time)
    if times.ndim != 1:
        msg = f"Expected 1D array for time, got {times.ndim}D"
        return Err(ValueError(msg))
    return create_time_series_core(prefix, [v.name for v in vtus], cast("A1[np.float64]", times))


def create_time_series_range(
    prefix: str,
    time: tuple[float, float, int],
    root: Path,
) -> Ok[TIME_SERIES] | Err:
    vtus = root.glob(f"{prefix}-*.vtu")
    times = np.linspace(time[0], time[1], time[2], dtype=np.float64)
    return create_time_series_core(prefix, [v.name for v in vtus], cast("A1[np.float64]", times))
