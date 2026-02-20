from typing import TYPE_CHECKING

from pytools.logging import get_logger
from pytools.result import Err, Ok

from ._impl_indexers import (
    ListIndexer,
    ListSubIndexer,
    RangeIndexer,
    RangeSubIndexer,
    TupleIndexer,
    ZeroIndexer,
)
from ._search import find_var_index, find_var_subindex
from .trait import IIndexIterator, SearchMode

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


__all__ = [
    "find_common_index",
    "find_common_subindex",
    "get_file_name_indexer",
]


def find_common_index(var: Sequence[str], root: Path | str | None = None) -> Ok[list[int]] | Err:
    match find_var_index(var[0], root):
        case Ok(indices):
            if len(indices) < 1:
                msg = f"No files found for {var[0]} in {root}\nNo variable outputed"
                return Err(ValueError(msg))
        case Err(e):
            return Err(e)
    return Ok(indices)


def find_common_subindex(
    var: Sequence[str],
    root: Path | str | None = None,
    index: Iterable[int] | None = None,
) -> Ok[dict[int, list[int]]] | Err:
    match find_var_subindex(var[0], root):
        case Ok(indices):
            common_keys = sorted(set(indices) & set(index)) if index else sorted(indices)
        case Err(e):
            return Err(e)
    return Ok({k: indices[k] for k in common_keys})


def _get_auto_indexer(
    variables: Sequence[str],
    root: Path | str | None,
    subindex: tuple[int, int, int] | None,
) -> Ok[IIndexIterator] | Err:
    match find_common_index(variables, root):
        case Ok(idx): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match subindex:
        case None:
            return Ok(ListIndexer(idx))
        case (int(), int(), int()):
            return Ok(ListSubIndexer(idx, subindex))


def _get_auto_subindexer(
    variables: Sequence[str],
    root: Path | str | None,
    main_index: tuple[int, int, int] | SearchMode | None,
) -> Ok[IIndexIterator] | Err:
    match main_index:
        case (int(), int(), int()):
            common_idx = range(*main_index)
        case _:
            common_idx = None
    match find_common_subindex(variables, root, common_idx):
        case Ok(subidx): ...  # fmt: skip
        case Err(e):
            return Err(e)
    return Ok(TupleIndexer(subidx))


def get_file_name_indexer(
    index: tuple[int, int, int] | SearchMode | None,
    subindex: tuple[int, int, int] | SearchMode | None,
    variables: Sequence[str],
    *,
    root: Path | str | None = None,
) -> Ok[IIndexIterator] | Err:
    if not variables:
        return Ok(ZeroIndexer())
    if isinstance(index, SearchMode) or isinstance(subindex, SearchMode):
        log = get_logger()
        log.info(
            f"<<< Variable index will be determined from the first variable: {variables[0]}",
        )
    match index, subindex:
        case None, _:
            return Ok(ZeroIndexer())
        case (int(), int(), int()), None:
            return Ok(RangeIndexer(index))
        case (int(), int(), int()), (int(), int(), int()):
            return Ok(RangeSubIndexer(index, subindex))
        case SearchMode(), None | (int(), int(), int()):
            return _get_auto_indexer(variables, root, subindex)
        case SearchMode() | (int(), int(), int()), SearchMode():
            return _get_auto_subindexer(variables, root, index)
