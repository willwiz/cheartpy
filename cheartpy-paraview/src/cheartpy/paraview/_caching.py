from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, overload

import numpy as np
from cheartpy.io import chread_d
from pytools.logging import ILogger, get_logger
from pytools.result import Err, Ok

from ._struct import ParaviewTopology, ProgramArgs, VariableCache, XMLDataInputs
from ._variable_getter import CheartVTUFormat

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from cheartpy.search import IIndexIterator
    from pytools.arrays import A2, DType

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
    fv = {k: v[i0] for k, v in inp.point_var.items()}
    fc = {k: v[i0] for k, v in inp.cell_var.items()}
    return Ok(VariableCache(top, i0, fx, fd, fv, fc, ftype, dtype))


@overload
def check_validate_v(v: None, time: int | str, backup: Path | None) -> None: ...
@overload
def check_validate_v(v: IFormattedName, time: int | str, backup: Path) -> Path: ...
@overload
def check_validate_v(
    v: IFormattedName | None, time: int | str, backup: Path | None
) -> Path | None: ...
def check_validate_v(v: IFormattedName | None, time: int | str, backup: Path | None) -> Path | None:
    if v is None:
        return v
    name = v[time]
    if name.is_file():
        return name
    log = get_logger()
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
    fx = check_validate_v(inp.space, time, cache.fx)
    fd = check_validate_v(inp.disp, time, cache.fd)
    fv = {k: check_validate_v(v, time, cache.fv[k]) for k, v in inp.point_var.items()}
    fc = {k: check_validate_v(v, time, cache.fc[k]) for k, v in inp.cell_var.items()}
    return VariableCache(cache.top, time, fx, fd, fv, fc, cache.ftype, cache.dtype)


class _TExportVariable[F: np.floating](NamedTuple):
    x: A2[F]
    v: Mapping[str, A2[F]]
    c: Mapping[str, A2[F]]


def get_arguments[F: np.floating, I: np.integer](
    inp: ProgramArgs, cache: VariableCache[F, I], indexer: IIndexIterator, *, log: ILogger
) -> Generator[XMLDataInputs[F, I]]:
    path_getter = CheartVTUFormat(inp.output_dir, inp.prefix)
    for t in indexer:
        cache = update_variable_cache(inp, t, cache, log=log)
        yield XMLDataInputs(
            prefix=inp.prefix,
            path=path_getter[t],
            time=t,
            top=cache.top,
            x=cache.fx,
            u=cache.fd,
            point_var=cache.fv,
            cell_var=cache.fc,
            compress=inp.compress,
            ftype=cache.ftype,
            dtype=cache.dtype,
        )


def get_xml_variables[F: np.floating, I: np.integer](
    xml: XMLDataInputs[F, I],
) -> _TExportVariable[F]:
    dtype = xml.ftype
    fx_data = chread_d(xml.x, dtype=dtype) if isinstance(xml.x, Path) else xml.top.x
    if xml.u is not None:
        fx_data = (fx_data + chread_d(xml.u, dtype=dtype)).astype(dtype)
    fv_data = {
        k: chread_d(v, dtype=dtype) if isinstance(v, Path) else v.astype(dtype)
        for k, v in xml.point_var.items()
    }
    fc_data = {
        k: chread_d(v, dtype=dtype) if isinstance(v, Path) else v.astype(dtype)
        for k, v in xml.cell_var.items()
    }
    return _TExportVariable(x=fx_data, v=fv_data, c=fc_data)
