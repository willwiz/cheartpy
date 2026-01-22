from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Unpack, cast

import numpy as np
from cheartpy.io.api import read_array_float
from cheartpy.search.api import get_var_index
from pytools.logging import BLogger
from pytools.result import Err, Ok

from ._headers import header_guard
from ._parser.time_parser import get_cmdline_args

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pytools.arrays import A1, DType

    from ._parser import TimeSeriesKwargs
    from ._trait import TIME_SERIES

__all__ = ["_create_time_series_file", "_create_time_series_range"]


CURRENT_VERSION: Final[str] = "1.0.0"
_H_STR_LEN_ = 30


def compose_time_header() -> Sequence[str]:
    return (
        header_guard(),
        "    Program for creating Paraview time series file (JSON).",
        "    This program is part of the CHeart project.",
        f"    Currently supporting version {CURRENT_VERSION}",
        "    Author: Will Zhang",
        "    Date: 1/20/2026",
        header_guard(),
    )


def format_input_info(prefix: str, root: Path) -> Sequence[str]:
    return (
        f"{'<<< VTU directory:':<{_H_STR_LEN_}} {root}",
        f"{'<<< Input file name prefix:':<{_H_STR_LEN_}} {prefix}",
        f"{'<<< Output file name:':<{_H_STR_LEN_}} {prefix + '.series'}\n",
    )


def create_time_series_json[F: np.floating](
    vtus: Iterable[Path | str], times: A1[F]
) -> TIME_SERIES:
    return {
        "file-series-version": CURRENT_VERSION,
        "files": [{"name": str(n), "time": float(t)} for n, t in zip(vtus, times, strict=False)],
    }


def _create_time_series_file[F: np.floating](
    vtus: Iterable[Path], idx: Sequence[int], time: str, *, dtype: DType[F] = np.float64
) -> Ok[tuple[Sequence[int], Iterable[Path], A1[F]]] | Err:
    time_array = read_array_float(time, dtype=dtype)
    match time_array.shape:
        case (int(),):
            time_array = cast("A1[F]", time_array)
        case _:
            msg = f"Expected 1D array for time, got {time_array.shape}"
            return Err(ValueError(msg))
    if len(time_array) <= max(idx):
        msg = f"Time array length {len(time_array)} must be > max index {max(idx)}"
        return Err(IndexError(msg))
    return Ok((idx, vtus, time_array[idx]))


def _create_time_series_range[F: np.floating](
    vtus: Iterable[Path],
    idx: Sequence[int],
    time_step: float,
    *,
    dtype: DType[F] = np.float64,
) -> Ok[tuple[Sequence[int], Iterable[Path], A1[F]]] | Err:
    nt = max(idx)
    time = np.arange(0, time_step * (nt + 1), time_step, dtype=dtype)
    return Ok((idx, vtus, time[idx]))


def create_time_series_core[F: np.floating](
    prefix: str,
    root: Path,
    time: float | str,
    *,
    dtype: DType[F] = np.float64,
) -> Ok[tuple[Sequence[int], Iterable[Path], A1[F]]] | Err:
    vtus = (v for v in root.glob(f"{prefix}-*.vtu"))
    match get_var_index((v.name for v in vtus), prefix, "vtu"):
        case Ok(idx):
            pass
        case Err(e):
            return Err(e)
    match time:
        case str():
            return _create_time_series_file(vtus, idx, time, dtype=dtype).next()
        case float() | int():
            return _create_time_series_range(vtus, idx, time, dtype=dtype).next()


def create_time_series_api(
    prefix: str,
    time: float | str,
    **kwargs: Unpack[TimeSeriesKwargs],
) -> Ok[None] | Err:
    root = kwargs.get("root", Path.cwd())
    dtype = kwargs.get("dtype", np.float64)
    log = BLogger(kwargs.get("log", "INFO"))
    log.disp(*compose_time_header())
    log.info(*format_input_info(prefix, root))
    log.disp("", header_guard())
    match create_time_series_core(prefix, root, time, dtype=dtype):
        case Ok((idx, vtus, time_array)):
            msg = (
                f"<<< Found {len(idx)} VTU files for time series, from:",
                f"        start = {idx[0]}, end = {idx[-1]}",
            )
            log.info(msg)
        case Err(e):
            return Err(e)
    time_series = create_time_series_json(vtus, time_array)
    with (root / (prefix + ".series")).open("w") as f:
        json.dump(time_series, f)
    return Ok(None)


def create_time_series_cli(cmdline: Sequence[str] | None = None) -> None:
    args = get_cmdline_args(cmdline)
    create_time_series_api(args.prefix, args.time, root=args.root).unwrap()
