import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from pytools.logging import get_logger
from pytools.result import Err, Ok, Result

from ._impl_indexers import (
    ListIndexer,
    RangeIndexer,
    RangeSubIndexer,
    TupleIndexer,
    TupleSubIndexer,
    ZeroIndexer,
)
from .trait import DynamicFile, FileVariable, IIndexIterator, SearchMode, StaticFile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pytools.arrays import T3


def filter_index(files: Sequence[Path], prefix: str, extension: str) -> set[int]:
    """Return the set of indices found in the list of files.

    Parameters
    ----------
    files : Sequence[Path]
        The list of files to search for indices.
    prefix : str
        The prefix of the variable to search for.
    extension : str
        The extension of the variable to search for.

    Returns
    -------
    set[int]
        The set of indices found in the list of files.

    """
    pattern = re.compile(rf"{prefix}-(?P<index>\d+)\.{extension}")
    return {int(m.group("index")) for f in files if (m := pattern.fullmatch(f.name))}


def filter_subindex(files: Sequence[Path], prefix: str, extension: str) -> set[tuple[int, int]]:
    """Return the dictionary of indices and subindices found in the list of files.

    Parameters
    ----------
    files : Sequence[Path]
        The list of files to search for indices.
    prefix : str
        The prefix of the variable to search for.
    extension : str
        The extension of the variable to search for.

    Returns
    -------
    dict[int, list[int]]
        The dictionary of indices and subindices found in the list of files.

    """
    pattern = re.compile(rf"{prefix}-(?P<index>\d+)\.(?P<subindex>\d+)\.{extension}")
    return {
        (int(m.group("index")), int(m.group("subindex")))
        for f in files
        if (m := pattern.fullmatch(f.name))
    }


def _determine_file_type(
    name: Path, ext: str, index: set[int], subindex: set[tuple[int, int]]
) -> Result[FileVariable]:
    if len(index) + len(subindex) == 1:
        match (index | subindex).pop():
            case int(i):
                return Ok(
                    FileVariable(StaticFile(name.parent / f"{name.stem}-{i}{ext}"), index, subindex)
                )
            case (i, j):
                return Ok(
                    FileVariable(
                        StaticFile(name.parent / f"{name.stem}-{i}.{j}{ext}"), index, subindex
                    )
                )
    if not index and not subindex:
        msg = f"No files found for {name} with extension {ext}"
        return Err(ValueError(msg))
    return Ok(FileVariable(DynamicFile(name.parent, name.stem, ext), index, subindex))


def get_file_type(name: Path | str, root: Path) -> Result[FileVariable]:
    """Return the format of the variables found.

    When searching, prefers:
    1. Single file if it exists.
    2. Variable in parent if a path is given.
    3. Variable in root if an input directory is given.

    if multiple file types are found, prefers:
    1. .D
    2. .D.gz
    3. .res2

    Parameters
    ----------
    name : Path | str
        The name of the variable to search for. An existing file or prefix for the variable, i.e.,
        before dash [`-`]
    root : Path
        The backup root directory to search for the variable.

    Returns
    -------
    fname: VariableType
        The format of the variable found.
    files: list[Path]
        The list of files found for the variable.

    """
    name = Path(name)
    if name.is_file():
        return Ok(FileVariable(StaticFile(name), set(), set()))
    if str(name) != name.name:
        for ext in [".D", ".D.gz", ".res2"]:
            if files := sorted(name.parent.glob(f"{name.name}-*{ext}")):
                index = filter_index(files, name.name, ext.lstrip("."))
                subindex = filter_subindex(files, name.name, ext.lstrip("."))
                return _determine_file_type(name, ext, index, subindex).next()
    for ext in [".D", ".D.gz", ".res2"]:
        if files := sorted(root.glob(f"{name.name}-*{ext}")):
            index = filter_index(files, name.name, ext.lstrip("."))
            subindex = filter_subindex(files, name.name, ext.lstrip("."))
            return _determine_file_type(name, ext, index, subindex).next()
    msg = (
        f"{name} not found as:\n"
        f"    {name}\n"
        f"or  {name.parent / f'{name.name}-*{ext}'}\n"
        f"or  {root / f'{name.name}-*{ext}'}"
    )
    return Err(ValueError(msg))


def build_subindex_mapping(
    var: Mapping[str, FileVariable],
) -> Mapping[int, Sequence[int]]:
    subindex_mapping: dict[int, set[int]] = defaultdict(set)
    for v in var.values():
        for idx, subidx in v.subindices:
            subindex_mapping[idx].add(subidx)
    return {k: sorted(v) for k, v in subindex_mapping.items()}


def create_range_indexer(
    var: Mapping[str, FileVariable],
    index: T3[int],
    subindexer: T3[int] | SearchMode,
) -> Result[IIndexIterator]:
    if not var:
        return Ok(ZeroIndexer())
    match subindexer:
        case SearchMode.none:
            return Ok(RangeIndexer(index, inclusive=True))
        case tuple():
            return Ok(RangeSubIndexer(index, subindexer, inclusive=True))
        case SearchMode.auto:
            subindex_map = build_subindex_mapping(var)
            indexer = RangeIndexer(index, inclusive=True)
            return Ok(TupleIndexer({i: subindex_map[i] for i in indexer}))


def create_null_indexer(
    var: Mapping[str, FileVariable],
    subindex: T3[int] | SearchMode,
) -> Result[IIndexIterator]:
    """Create an indexer assuming no index should be generated."""
    if not var:
        return Ok(ZeroIndexer())
    match subindex:
        case SearchMode.none:
            msg = "No index mode or subindex mode specified"
            return Err(ValueError(msg))
        case tuple():
            subindex_map = build_subindex_mapping(var)
            subindex_range = range(*subindex)
            return Ok(
                TupleSubIndexer(
                    {k: [v[i] for i in subindex_range] for k, v in subindex_map.items()}
                )
            )
        case SearchMode.auto:
            subindex_map = build_subindex_mapping(var)
            return Ok(TupleSubIndexer(subindex_map))


def create_auto_indexer(
    var: Mapping[str, FileVariable],
    sub_index: T3[int] | SearchMode,
) -> Result[IIndexIterator]:
    if not var:
        return Ok(ZeroIndexer())
    index = sorted(set[int].union(*(v.indices for v in var.values())))
    match sub_index:
        case SearchMode.none:
            return Ok(ListIndexer(index))
        case tuple():
            return Ok(RangeSubIndexer((0, 0, 1), sub_index, inclusive=True))
        case SearchMode.auto:
            subindex_map = build_subindex_mapping(var)
            return Ok(TupleIndexer(subindex_map))


def validate_indexer(
    var: Mapping[str, FileVariable], indexer: IIndexIterator
) -> Result[IIndexIterator]:
    var = {k: v for k, v in var.items() if v.fname.is_dynamic}
    first = next(iter(indexer))
    initial_file_check = {
        k for k, v in var.items() if not (first in v.indices or first in v.subindices)
    }
    if initial_file_check:
        msg = f"Variables {initial_file_check} don't have the inital index file {first}."
        return Err(ValueError(msg))
    file_completion = {
        k: [i for i in indexer if not (i in v.indices or i in v.subindices)] for k, v in var.items()
    }
    log = get_logger()
    for k, v in file_completion.items():
        if v:
            log.warn(
                "Variable is missing files, and will use previous", variable=k, missing_files=v
            )
    return Ok(indexer)


def create_indexer(
    var: Mapping[str, FileVariable],
    index: T3[int] | SearchMode,
    sub_index: T3[int] | SearchMode,
) -> Result[IIndexIterator]:
    var = {k: v for k, v in var.items() if v.fname.is_dynamic}
    match index:
        case tuple():
            indexer = create_range_indexer(var, index, sub_index)
        case SearchMode.auto:
            indexer = create_auto_indexer(var, sub_index)
        case SearchMode.none:
            indexer = create_null_indexer(var, sub_index)
    return indexer.and_then(lambda x: validate_indexer(var, x)).next()
