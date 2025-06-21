from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["update_variable_cache"]

from typing import TYPE_CHECKING

from cheartpy.cheart_mesh.io import chread_d

from .struct import CheartTopology, ProgramArgs, VariableCache

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arraystubs import Arr2
    from pytools.logging.trait import ILogger

    from cheartpy.io.indexing.interfaces import IIndexIterator


def init_variable_cache(
    inp: ProgramArgs,
    indexer: IIndexIterator,
) -> VariableCache:
    i0 = next(iter(indexer))
    top = CheartTopology(inp.tfile, inp.bfile)
    fx = inp.space[i0]
    space = chread_d(fx)
    if inp.disp is None:
        fd = None
        disp = np.zeros_like(space)
    else:
        fd = inp.disp[i0]
        disp = chread_d(fd)
    x = space + disp
    fv: dict[str, Path] = dict.fromkeys(inp.var.keys(), Path())
    var: dict[str, Arr2[np.float64]] = {}
    for k, fn in inp.var.items():
        name = fn[i0]
        if name.exists():
            fv[k] = name
        else:
            msg = f"initial value for {k} = {name} does not exist"
            raise ValueError(msg)
        var[k] = chread_d(name)
    return VariableCache(top, i0, fx, fd, space, disp, x, fv, var)


def update_variable_cache(
    inp: ProgramArgs,
    time: int | str,
    cache: VariableCache,
    log: ILogger,
) -> tuple[Arr2[np.float64], Mapping[str, Arr2[np.float64]]]:
    if time == cache.t:
        log.debug(f"time point {time} did not change")
        return cache.x.astype(dtype=np.float64), {
            k: v.astype(dtype=np.float64) for k, v in cache.var.items()
        }
    fx = inp.space[time]
    update_space = fx != cache.space_i
    if update_space:
        log.debug(f"updating space to file {fx}")
        cache.space = chread_d(fx)
        cache.space_i = fx
    if inp.disp is None:
        update_disp = False
    else:
        fd = inp.disp[time]
        update_disp = fd != cache.disp_i
        if update_disp:
            log.debug(f"updating disp to file {fd}")
            cache.disp = chread_d(fd)
            cache.disp_i = fd
    match update_space, update_disp:
        case False, False:
            pass
        case True, False:
            cache.x = cache.space
        case _, True:
            cache.x = cache.space + cache.disp
    for k, var in inp.var.items():
        new_v = var[time]
        log.debug(f"updating var {k} to file {new_v} from {cache.var_i[k]}")
        if (cache.var_i[k] != new_v) and Path(new_v).exists():
            log.debug(f"updating var {k} to file {new_v}")
            cache.var[k] = chread_d(new_v)
            cache.var_i[k] = new_v
    return cache.x, cache.var
