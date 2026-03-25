from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pytools.logging import ILogger
from pytools.result import Err, Ok

from .trait import AUTO as AUTO
from .trait import IIndexIterator as IIndexIterator
from .trait import ProgramMode as ProgramMode
from .trait import SearchMode as SearchMode

def check_for_var_files(
    idx: IIndexIterator,
    *var: str,
    suffix: Literal[".D", ".D.gz"] = ".D",
    root: Path | str | None = None,
    log: ILogger | None = None,
) -> bool: ...
def find_common_index(
    var: Sequence[str], root: Path | str | None = None
) -> Ok[list[int]] | Err: ...
def find_common_subindex(
    var: Sequence[str], root: Path | str | None = None, index: Iterable[int] | None = None
) -> Ok[dict[int, list[int]]] | Err: ...
def find_var_index(prefix: str, root: Path | str | None) -> Ok[list[int]] | Err: ...
def find_var_subindex(prefix: str, root: Path | str | None) -> Ok[dict[int, list[int]]] | Err: ...
def get_file_name_indexer(
    index: tuple[int, int, int] | SearchMode | None,
    subindex: tuple[int, int, int] | SearchMode | None,
    variables: Sequence[str],
    *,
    root: Path | str | None = None,
) -> Ok[IIndexIterator] | Err: ...
def get_var_index(
    names: Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz", r"vtu"] = r"D",
) -> Ok[list[int]] | Err: ...
def get_var_index_all(
    names: Sequence[str] | Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz"] = r"D",
) -> list[str]: ...
def get_var_subindex(
    names: Iterable[str],
    prefix: str,
    suffix: Literal[r"D", r"D\.gz"] = r"D",
) -> Ok[dict[int, list[int]]] | Err: ...
