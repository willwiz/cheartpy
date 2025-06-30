__all__ = [
    "find_common_index",
    "find_common_subindex",
    "get_file_name_indexer",
]
from collections.abc import Mapping, Sequence
from pathlib import Path

from pytools.logging.api import BLogger
from pytools.logging.trait import ILogger

from .indexers import (
    ListIndexer,
    ListSubIndexer,
    RangeIndexer,
    RangeSubIndexer,
    TupleIndexer,
    ZeroIndexer,
)
from .interfaces import IIndexIterator, SearchMode
from .search import find_var_index, find_var_subindex


def find_common_index(
    var: Sequence[str],
    root: Path | str | None = None,
    log: ILogger | None = None,
) -> Sequence[int]:
    if log is None:
        log = BLogger("WARN")
    indices = find_var_index(var[0], root, log)
    if len(indices) < 1:
        log.warn(f"No files found for {var[0]} in {root}\nNo variable outputed")
        raise ValueError
    return indices


def find_common_subindex(
    var: Sequence[str],
    root: Path | str | None = None,
    index: Sequence[int] | None = None,
    log: ILogger | None = None,
) -> Mapping[int, Sequence[int]]:
    if log is None:
        log = BLogger("WARN")
    indices = find_var_subindex(var[0], root, log)
    common_keys = sorted(set(indices) & set(index) if index else list(indices))
    if len(common_keys) < 1:
        log.warn(f"No files found for {var[0]} in {root}\nNo variable outputed")
        raise ValueError
    return {k: indices[k] for k in common_keys}


def get_file_name_indexer(
    index: tuple[int, int, int] | SearchMode,
    subindex: tuple[int, int, int] | SearchMode,
    variables: Sequence[str],
    root: Path | str | None = None,
    log: ILogger | None = None,
) -> IIndexIterator:
    if log is None:
        log = BLogger("WARN")
    if (index is SearchMode.auto) or (subindex is SearchMode.auto):
        log.info(
            f"Variable index will be determined from the first variable: {variables[0]}",
        )
    match index, subindex:
        case SearchMode.none, _:
            indexer = ZeroIndexer()
        case SearchMode.auto, SearchMode.none:
            indexer = ListIndexer(find_common_index(variables, root, log))
        case SearchMode.auto, SearchMode.auto:
            indexer = TupleIndexer(find_common_subindex(variables, root, log=log))
        case SearchMode.auto, (int(), int(), int()):
            indexer = ListSubIndexer(find_common_index(variables, root, log), subindex)
        case (int(), int(), int()), SearchMode.none:
            indexer = RangeIndexer(index)
        case (int(start), int(end), int(step)), SearchMode.auto:
            indexer = TupleIndexer(
                find_common_subindex(variables, root, range(start, end, step), log=log),
            )
        case (int(), int(), int()), (int(), int(), int()):
            indexer = RangeSubIndexer(index, subindex)
    return indexer
