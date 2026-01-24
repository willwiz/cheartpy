from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, overload

import numpy as np
from cheartpy.io.api import chread_d
from cheartpy.paraview._variable_getter import CheartVTUFormat
from pytools.result import Err, Ok

from ._struct import ParaviewTopology, ProgramArgs, VariableCache, XMLDataInputs

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from cheartpy.search.trait import IIndexIterator
    from pytools.arrays import A2, DType
    from pytools.logging import ILogger

    from ._trait import IFormattedName

__all__ = ["init_variable_cache", "update_variable_cache"]


def init_variable_cache[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    indexer: IIndexIterator,
    *,
    ftype: DType[F] = np.float64,
    dtype: DType[I] = np.intp,
) -> Ok[VariableCache[F, I]] | Err:
    """Initialize the variable cache.

    If new variable cannot be found then this is the backup.
    The first time index always exists, as it is checked in `process_cmdline_args`.

    """
    i0 = next(iter(indexer))
    space = chread_d(inp.xfile, dtype=ftype)
    top = ParaviewTopology(space, inp.tfile, inp.bfile, dtype=dtype)
    fx = None if inp.space is None else inp.space[i0]
    fd = None if inp.disp is None else inp.disp[i0]
    fv = {k: v[i0] for k, v in inp.var.items()}
    return Ok(VariableCache(top, i0, fx, fd, fv, ftype, dtype))


@overload
def check_validate_v(v: None, time: int | str, backup: Path | None, *, log: ILogger) -> None: ...
@overload
def check_validate_v(v: IFormattedName, time: int | str, backup: Path, *, log: ILogger) -> Path: ...
@overload
def check_validate_v(
    v: IFormattedName | None, time: int | str, backup: Path | None, *, log: ILogger
) -> Path | None: ...
def check_validate_v(
    v: IFormattedName | None, time: int | str, backup: Path | None, *, log: ILogger
) -> Path | None:
    if v is None:
        return v
    name = v[time]
    if name.is_file():
        return name
    msg = f"disp file (t = {time}) = {name} does not exist.\n"
    msg += f"using previous step ({backup})"
    log.warn(msg)
    return backup


def update_variable_cache[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    time: int | str,
    cache: VariableCache[F, I],
    log: ILogger,
) -> VariableCache[F, I]:
    if time == cache.time:
        log.debug(f"time point {time} did not change")
        return cache
    fx = check_validate_v(inp.space, time, cache.fx, log=log)
    fd = check_validate_v(inp.disp, time, cache.fd, log=log)
    fv = {k: check_validate_v(v, time, cache.fv[k], log=log) for k, v in inp.var.items()}
    return VariableCache(cache.top, time, fx, fd, fv, cache.ftype, cache.dtype)


class _TExportVariable[F: np.floating](NamedTuple):
    x: A2[F]
    v: Mapping[str, A2[F]]


def get_arguments[F: np.floating, I: np.integer](
    inp: ProgramArgs, cache: VariableCache[F, I], indexer: IIndexIterator, *, log: ILogger
) -> Generator[tuple[tuple[Path, XMLDataInputs[F, I]], Mapping[str, Any]]]:
    path_getter = CheartVTUFormat(inp.output_dir, inp.prefix)
    for t in indexer:
        cache = update_variable_cache(inp, t, cache, log=log)
        args = XMLDataInputs(
            prefix=inp.prefix,
            time=t,
            top=cache.top,
            x=cache.fx,
            u=cache.fd,
            var=cache.fv,
            compress=inp.compress,
        )
        yield (path_getter[t], args), {}


def get_variables[F: np.floating, I: np.integer](
    top: ParaviewTopology[F, I],
    fx: Path | None,
    fu: Path | None,
    fv: Mapping[str, Path] | Mapping[str, A2[np.floating]],
    *,
    dtype: DType[F] = np.float64,
) -> _TExportVariable[F]:
    fx_data = chread_d(fx, dtype=dtype) if isinstance(fx, Path) else top.x
    if fu is not None:
        fx_data = (fx_data + chread_d(fu, dtype=dtype)).astype(dtype)
    fv_data = {
        k: chread_d(v, dtype=dtype) if isinstance(v, Path) else v.astype(dtype)
        for k, v in fv.items()
    }
    return _TExportVariable(x=fx_data, v=fv_data)
