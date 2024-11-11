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
    indices = find_var_index(var[0], root)
    if len(indices) < 1:
        LOG.warn(f"No files found for {var[0]}\nNo variable outputed")
    return indices


def find_common_subindex(
    var: list[str],
    root: str | None = None,
    index: Sequence[int] | None = None,
    LOG: ILogger = BLogger("WARN"),
):
    indices = find_var_subindex(var[0], root)
    # common_keys = set(indices).intersection(*indices.values())
    # if index:
    common_keys = sorted(set(indices) & set(index) if index else list(indices))
    # common_index: dict[int, list[int]] = dict()
    # for k in common_keys:
    #     common_index[k] = sorted(
    #         set(indices[var[0]]).intersection(*[indices[v] for v in var])
    #     )
    if len(common_keys) < 1:
        LOG.warn(f"No files found for {var[0]}\nNo variable outputed")
    return {k: indices[k] for k in common_keys}


def get_file_name_indexer(
    index: tuple[int, int, int] | SearchMode,
    subindex: tuple[int, int, int] | SearchMode,
    vars: list[str],
    root: str | None = None,
    LOG: ILogger = BLogger("WARN"),
) -> IIndexIterator:
    if (index is SearchMode.auto) or (subindex is SearchMode.auto):
        LOG.info(
            f"Variable index will be determined from the first variable, {vars[0]}"
        )
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
