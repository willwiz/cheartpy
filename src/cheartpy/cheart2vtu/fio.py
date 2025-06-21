__all__ = ["update_variable_cache"]
import os
from typing import Mapping

from ..cheart_mesh.io import *
from ..tools.basiclogging import BLogger, ILogger
from ..var_types import *
from .interfaces import *


def update_variable_cache(
    inp: ProgramArgs,
    time: int | str,
    cache: VariableCache,
    LOG: ILogger = BLogger("NULL"),
) -> tuple[Mat[f64], Mapping[str, Mat[f64]]]:
    if time == cache.t:
        LOG.debug(f"time point {time} did not change")
        return cache.x, cache.var
    fx = inp.space[time]
    update_space = fx != cache.space_i
    if update_space:
        LOG.debug(f"updating space to file {fx}")
        cache.space = chread_d(fx)
        cache.space_i = fx
    if inp.disp is None:
        update_disp = False
    else:
        fd = inp.disp[time]
        update_disp = fd != cache.disp_i
        if update_disp:
            LOG.debug(f"updating disp to file {fd}")
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
        LOG.debug(f"updating var {k} to file {new_v} from {cache.var_i[k]}")
        if (cache.var_i[k] != new_v) and os.path.isfile(new_v):
            LOG.debug(f"updating var {k} to file {new_v}")
            cache.var[k] = chread_d(new_v)
            cache.var_i[k] = new_v
    return cache.x, cache.var
