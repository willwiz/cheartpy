from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pytools.logging.trait import ILogger

    from .trait import IIndexIterator

__all__ = ["check_for_var_files"]


def check_for_var_files(
    idx: IIndexIterator,
    *var: str,
    suffix: Literal[".D", ".D.gz"] = ".D",
    root: Path | str | None = None,
    log: ILogger | None = None,
) -> bool:
    okay: bool = True
    log.debug(f"{list(idx)}") if log else ...
    root = Path(root) if root else Path()
    for v in var:
        for i in idx:
            if (root / f"{v}-{i}.{suffix}").is_file():
                continue
            okay = False
            log.error(f"{v}-{i}.{suffix} could not be found") if log else ...
    return okay
