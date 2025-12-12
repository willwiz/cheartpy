import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pytools.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


__all__ = [
    "find_var_index",
    "find_var_subindex",
    "get_var_index",
    "get_var_index_all",
    "get_var_subindex",
]
DFILE_TEMP = re.compile(r"(^.*)-(\d+|\d+\.\d+)\.(D|D\.gz)")


def get_var_index(
    names: Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz", r"vtu"] = r"D",
) -> Ok[list[int]] | Err:
    """Extract variable indices from a list of file names."""
    if r".\\" in prefix:
        msg = f"Prefix {prefix} should not contain '.\\'"
        return Err(ValueError(msg))
    p = re.compile(rf"{prefix}-(\d+)\.{suffix}")
    matches = [p.fullmatch(s) for s in names]
    if not any(matches):
        msg = f"No matching variable files found with prefix {prefix} and suffix {suffix}"
        return Err(ValueError(msg))
    return Ok(sorted([int(m.group(1)) for m in matches if m]))


def get_var_subindex(
    names: Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz"] = r"D",
) -> Ok[dict[int, list[int]]] | Err:
    if r".\\" in prefix:
        msg = f"Prefix {prefix} should not contain '.\\'"
        return Err(ValueError(msg))
    p = re.compile(rf"{prefix}-(\d+)\.(\d+)\.{suffix}")
    matches = [p.fullmatch(s) for s in names]
    if not any(matches):
        msg = f"No matching variable files found with prefix {prefix} and suffix {suffix}"
        return Err(ValueError(msg))
    matches = sorted([m.groups() for m in matches if m])
    index_lists: dict[int, list[int]] = defaultdict(list)
    for k, i in matches:
        index_lists[int(k)].append(int(i))
    return Ok(index_lists)


def get_var_index_all(
    names: Sequence[str] | Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz"] = r"D",
) -> list[str]:
    p = re.compile(rf"{prefix}-(\d+|\d+.\d+).{suffix}")
    matches = [p.fullmatch(s) for s in names]
    return [m.group(1) for m in matches if m]


def find_var_index(prefix: str, root: Path | str | None) -> Ok[list[int]] | Err:
    root = Path(root) if root else Path()
    var, suffix = root.glob(f"{prefix}-*.D"), r"D"
    var = list(var)
    if not any(var):
        var, suffix = root.glob(f"{prefix}-*.D.gz"), r"D\.gz"
    return get_var_index([v.name for v in var], prefix, suffix)


def find_var_subindex(prefix: str, root: Path | str | None) -> Ok[dict[int, list[int]]] | Err:
    root = Path(root) if root else Path()
    var, suffix = root.glob(f"{prefix}-*.D"), r"D"
    var = list(var)
    if not any(var):
        var, suffix = root.glob(f"{prefix}-*.D.gz"), r"D\.gz"
    return get_var_subindex([v.name for v in var], prefix, suffix)
