import os
from typing import Literal
from ...tools.basiclogging import *
from ...tools.path_tools import path
from .interfaces import IIndexIterator


def check_for_var_files(
    idx: IIndexIterator,
    *var: str,
    suffix: Literal[".D", ".D.gz"] = ".D",
    root: str | None = None,
    LOG: ILogger = BLogger("ERROR"),
) -> bool:
    okay: bool = True
    LOG.debug(f"{list(idx)}")
    for v in var:
        for i in idx:
            if not os.path.isfile(path(root, f"{v}-{i}.{suffix}")):
                okay = False
                LOG.error(f"{v}-{i}.{suffix} could not be found")
    return okay
