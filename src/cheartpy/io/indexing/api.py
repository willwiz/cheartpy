__all__ = [
    "find_common_index",
    "find_common_subindex",
    "get_file_name_indexer",
]
from typing import Sequence
from ...tools.basiclogging import BLogger, ILogger
from ...var_types import *
from .interfaces import *
from .indexers import *
from .search import *


def find_common_index(
    var: list[str],
    root: str | None = None,
    LOG: ILogger = BLogger("WARN"),
):
    indices = {v: set(find_var_index(v, root)) for v in var}
    for k, v in indices.items():
        LOG.warn(f"No files found for {k}\nNo variable outputed") if len(v) < 1 else ...
    sets = indices[var[0]].intersection(*indices.values())
    return sorted(sets)


def find_common_subindex(
    var: list[str],
    root: str | None = None,
    index: Sequence[int] | None = None,
    LOG: ILogger = BLogger("WARN"),
):
    indices = {v: find_var_subindex(v, root) for v in var}
    common_keys = set(indices[var[0]]).intersection(*indices.values())
    if len(common_keys) < 1:
        LOG.warn(f"No files found variable sharing common indices")
    if index:
        common_keys = common_keys & set(index)
    common_index: dict[int, list[int]] = dict()
    for k in common_keys:
        common_index[k] = sorted(
            set(indices[var[0]]).intersection(*[indices[v] for v in var])
        )
        LOG.warn(f"No files at time step {k}") if len(common_index[k]) < 1 else ...
    return common_index


def get_file_name_indexer(
    index: tuple[int, int, int] | SearchMode,
    subindex: tuple[int, int, int] | SearchMode,
    vars: list[str],
    root: str | None = None,
    LOG: ILogger = BLogger("WARN"),
) -> IIndexIterator:
    match index, subindex:
        case SearchMode.none, _:
            return ZeroIndexer()
        case SearchMode.auto, SearchMode.none:
            return ListIndexer(find_common_index(vars, root, LOG))
        case SearchMode.auto, SearchMode.auto:
            return TupleIndexer(find_common_subindex(vars, root, LOG=LOG))
        case SearchMode.auto, (int(), int(), int()):
            return ListSubIndexer(find_common_index(vars, root, LOG), subindex)
        case (int(), int(), int()), SearchMode.none:
            return RangeIndexer(index)
        case (int(), int(), int()), SearchMode.auto:
            indicies = range(*index)
            return TupleIndexer(find_common_subindex(vars, root, indicies, LOG=LOG))
        case (int(), int(), int()), (int(), int(), int()):
            return RangeSubIndexer(index, subindex)
