__all__ = ["create_time_series_file", "create_time_series_range"]
from collections.abc import Sequence
from pathlib import Path
from typing import Final, ReadOnly, TypedDict, cast

import numpy as np
from arraystubs import Arr1

from cheartpy.io.indexing.search import get_var_index
from cheartpy.io.raw_io import read_array_float

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


def create_json(vtus: Sequence[str], times: Arr1[np.float64]) -> TIME_SERIES:
    return {
        "file-series-version": _CURRENT_VERSION,
        "files": [{"name": n, "time": t} for n, t in zip(vtus, times, strict=False)],
    }


def create_time_series_core(
    prefix: str,
    vtus: Sequence[str],
    times: Arr1[np.float64],
) -> TIME_SERIES:
    idx = get_var_index(vtus, prefix, "vtu")
    times = times[idx]
    return create_json(vtus, times)


def create_time_series_file(prefix: str, time: str, root: Path) -> TIME_SERIES:
    vtus = root.glob(f"{prefix}-*.vtu")
    times = read_array_float(time)
    if times.ndim != 1:
        msg = f"Expected 1D array for time, got {times.ndim}D"
        raise ValueError(msg)
    return create_time_series_core(prefix, [v.name for v in vtus], cast("Arr1[np.float64]", times))


def create_time_series_range(
    prefix: str,
    time: tuple[float, float, int],
    root: Path,
) -> TIME_SERIES:
    vtus = root.glob(f"{prefix}-*.vtu")
    times = np.linspace(time[0], time[1], time[2], dtype=np.float64)
    return create_time_series_core(prefix, [v.name for v in vtus], cast("Arr1[np.float64]", times))
