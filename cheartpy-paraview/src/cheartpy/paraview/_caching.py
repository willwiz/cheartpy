from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io.api import chread_d, chread_d_utf
from pytools.result import Err, Ok

from .struct import CheartTopology, ProgramArgs, VariableCache

if TYPE_CHECKING:
    from cheartpy.search.trait import IIndexIterator
    from pytools.arrays import A2, DType
    from pytools.logging.trait import ILogger

__all__ = ["init_variable_cache", "update_variable_cache"]


def init_variable_cache[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    indexer: IIndexIterator,
    itype: DType[I] = np.intc,
    ftype: DType[F] = np.float64,
) -> Ok[VariableCache[F, I]] | Err:
    i0 = next(iter(indexer))
    top = CheartTopology(inp.tfile, inp.bfile, dtype=itype)
    fx = inp.xfile[i0]
    space = chread_d(fx, dtype=ftype)
    if inp.disp is None:
        fd = None
        disp: A2[F] = np.zeros_like(space, dtype=ftype)
    else:
        fd = inp.disp[i0]
        disp = chread_d(fd, dtype=ftype)
    x = (space + disp).astype(ftype, copy=False)
    fv: dict[str, Path] = dict.fromkeys(inp.var.keys(), Path())
    var: dict[str, A2[F]] = {}
    for k, fn in inp.var.items():
        name = fn[i0]
        if name.is_file():
            fv[k] = name
        else:
            msg = f"initial value for {k} = {name} does not exist"
            return Err(ValueError(msg))
        var[k] = chread_d(name, dtype=ftype)
    return Ok(VariableCache(top, i0, fx, fd, space, disp, x, fv, var))


def update_variable_cache[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    time: int | str,
    cache: VariableCache[F, I],
    log: ILogger,
) -> tuple[A2[F], dict[str, A2[F]]]:
    ftype = cache.space.dtype.type
    if time == cache.t:
        log.debug(f"time point {time} did not change")
        return cache.x, cache.var
    fx = inp.xfile[time]
    update_space = fx != cache.space_i
    if update_space:
        log.debug(f"updating space to file {fx}")
        cache.space = chread_d_utf(fx, dtype=ftype)
        cache.space_i = fx
    if inp.disp is None:
        update_disp = False
    else:
        fd = inp.disp[time]
        update_disp = fd != cache.disp_i
        if update_disp:
            log.debug(f"updating disp to file {fd}")
            cache.disp = chread_d(fd, dtype=ftype)
            cache.disp_i = fd
    match update_space, update_disp:
        case False, False:
            pass
        case True, False:
            cache.x = cache.space
        case _, True:
            cache.x = (cache.space + cache.disp).astype(ftype, copy=False)
    for k, var in inp.var.items():
        new_v = var[time]
        log.debug(f"updating var {k} to file {new_v} from {cache.var_i[k]}")
        if (cache.var_i[k] != new_v) and new_v.is_file():
            log.debug(f"updating var {k} to file {new_v}")
            cache.var[k] = chread_d(new_v, dtype=ftype)
            cache.var_i[k] = new_v
    return cache.x, cache.var
