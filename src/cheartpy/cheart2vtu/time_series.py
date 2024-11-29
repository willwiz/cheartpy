# encoding: utf-8
__all__ = ["create_time_series_file", "create_time_series_range"]
import numpy as np
from glob import glob
from typing import Final, ReadOnly, TypedDict
from ..io.indexing.search import get_var_index
from ..io.raw_io import read_array_float
from ..var_types import Vec, f64


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


def create_json(vtus: list[str], times: Vec[f64]) -> TIME_SERIES:
    return {
        "file-series-version": _CURRENT_VERSION,
        "files": [{"name": n, "time": t} for n, t in zip(vtus, times)],
    }


def create_time_series_core(
    prefix: str, vtus: list[str], times: Vec[f64]
) -> TIME_SERIES:
    idx = get_var_index(vtus, prefix, "vtu")
    times = times[idx]
    return create_json(vtus, times)


def create_time_series_file(prefix: str, time: str):
    vtus = glob(f"{prefix}-*.vtu")
    times = read_array_float(time)
    return create_time_series_core(prefix, vtus, times)


def create_time_series_range(prefix: str, time: tuple[float, float, int]):
    vtus = glob(f"{prefix}-*.vtu")
    times = np.linspace(time[0], time[1], time[2], dtype=np.float64)
    return create_time_series_core(prefix, vtus, times)
