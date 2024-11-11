__all__ = [
    "get_var_index",
    "get_var_index_all",
    "get_var_subindex",
    "find_var_index",
    "find_var_subindex",
]
import re
from glob import glob
from collections import defaultdict
from typing import Literal

DFILE_TEMP = re.compile(r"(^.*)-(\d+|\d+\.\d+)\.(D|D\.gz)")


def get_var_index(
    names: list[str], prefix: str, suffix: Literal[r"D", r"D\.gz"] = r"D"
) -> list[int]:
    p = re.compile(rf"{prefix}-(\d+)\.{suffix}")
    matches = [p.fullmatch(s) for s in names]
    return sorted([int(m.group(1)) for m in matches if m])


def get_var_subindex(
    names: list[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz"] = r"D",
) -> dict[int, list[int]]:
    p = re.compile(rf"{prefix}-(\d+)\.(\d+).{suffix}")
    matches = [p.fullmatch(s) for s in names]
    matches = sorted([m.groups() for m in matches if m])
    index_lists: dict[int, list[int]] = defaultdict(list)
    for k, i in matches:
        index_lists[int(k)].append(int(i))
    return index_lists


def get_var_index_all(
    names: list[str], prefix: str, suffix: Literal[r"D", r"D\.gz"] = r"D"
) -> list[str]:
    p = re.compile(rf"{prefix}-(\d+|\d+.\d+).{suffix}")
    matches = [p.fullmatch(s) for s in names]
    return [m.group(1) for m in matches if m]


def find_var_index(prefix: str, root: str | None = None):
    var, suffix = glob(f"{prefix}-*.D", root_dir=root), r"D"
    if len(var) == 0:
        var, suffix = glob(f"{prefix}-*.D", root_dir=root), r"D\.gz"
    return get_var_index(var, prefix, suffix)


def find_var_subindex(prefix: str, root: str | None = None):
    var, suffix = glob(f"{prefix}-*.D", root_dir=root), r"D"
    if len(var) == 0:
        var, suffix = glob(f"{prefix}-*.D", root_dir=root), r"D\.gz"
    return get_var_subindex(var, prefix, suffix)
