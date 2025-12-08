__all__ = [
    "find_var_index",
    "find_var_subindex",
    "get_var_index",
    "get_var_index_all",
    "get_var_subindex",
]
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pytools.logging.trait import ILogger

DFILE_TEMP = re.compile(r"(^.*)-(\d+|\d+\.\d+)\.(D|D\.gz)")


def get_var_index(
    names: Sequence[str] | Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz", r"vtu"] = r"D",
) -> list[int]:
    """Extract variable indices from a list of file names."""
    if r".\\" in prefix:
        msg = f"Prefix {prefix} should not contain '.\\'"
        raise ValueError(msg)
    p = re.compile(rf"{prefix}-(\d+)\.{suffix}")
    matches = [p.fullmatch(s) for s in names]
    return sorted([int(m.group(1)) for m in matches if m])


def get_var_subindex(
    names: Sequence[str] | Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz"] = r"D",
) -> dict[int, list[int]]:
    if r".\\" in prefix:
        msg = f"Prefix {prefix} should not contain '.\\'"
        raise ValueError(msg)
    p = re.compile(rf"{prefix}-(\d+)\.(\d+)\.{suffix}")
    matches = [p.fullmatch(s) for s in names]
    matches = sorted([m.groups() for m in matches if m])
    index_lists: dict[int, list[int]] = defaultdict(list)
    for k, i in matches:
        index_lists[int(k)].append(int(i))
    return index_lists


def get_var_index_all(
    names: Sequence[str] | Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz"] = r"D",
) -> list[str]:
    p = re.compile(rf"{prefix}-(\d+|\d+.\d+).{suffix}")
    matches = [p.fullmatch(s) for s in names]
    return [m.group(1) for m in matches if m]


def find_var_index(prefix: str, root: Path | str | None, log: ILogger) -> list[int]:
    root = Path(root) if root else Path()
    log.debug(f"Searching for files with prefix: {prefix} in {root=}")
    var, suffix = root.glob(f"{prefix}-*.D"), r"D"
    var = list(var)
    if not any(var):
        var, suffix = root.glob(f"{prefix}-*.D.gz"), r"D\.gz"
    return get_var_index([v.name for v in var], prefix, suffix)


def find_var_subindex(
    prefix: str,
    root: Path | str | None,
    log: ILogger,
) -> dict[int, list[int]]:
    root = Path(root) if root else Path()
    log.debug(f"Searching for files with prefix: {prefix} in {root=}")
    var, suffix = root.glob(f"{prefix}-*.D"), r"D"
    var = list(var)
    if not any(var):
        var, suffix = root.glob(f"{prefix}-*.D.gz"), r"D\.gz"
    return get_var_subindex([v.name for v in var], prefix, suffix)
