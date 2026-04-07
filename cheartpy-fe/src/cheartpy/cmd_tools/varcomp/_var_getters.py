from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from cheartpy.search import get_var_index
from pytools.result import Err, Ok, Result

from ._traits import IVariableGenerator

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator


class VariableGeneratorConst(IVariableGenerator):
    __slots__ = ["_idx", "_path"]

    _path: Final[Path]
    _idx: Final[Iterable[int] | Iterable[str]]

    def __init__(self, prefix: str, root: Path, idx: Iterable[int] | Iterable[str]) -> None:
        self._path = root / prefix
        self._idx = idx

    def __iter__(self) -> Iterator[Path]:
        for _ in self._idx:
            yield self._path


class VariableGenerator(IVariableGenerator):
    __slots__ = ["_idx", "_prefix", "_root"]

    _prefix: Final[str]
    _root: Final[Path]
    _idx: Final[Iterable[int] | Iterable[str]]

    def __init__(self, prefix: str, root: Path, idx: Iterable[int] | Iterable[str]) -> None:
        self._prefix = prefix
        self._root = root
        self._idx = idx

    def __iter__(self) -> Iterator[Path]:
        for i in self._idx:
            yield self._root / f"{self._prefix}-{i}.D"


class InputVariable(NamedTuple):
    prefix: str
    root: Path
    idx: Iterable[int]


def _parse_const_variable(var: str, root: Path) -> Result[InputVariable]:
    if (root / var).is_file():
        return Ok(InputVariable(prefix=var, root=root, idx=[]))
    if Path(var).is_file():
        return Ok(InputVariable(prefix=var, root=Path.cwd(), idx=[]))
    msg = f"Variable {var} could not be found in {root}. Did you mean to add a wildcard *?"
    return Err(ValueError(msg))


def _parse_indexed_variable(var: str, root: Path) -> Result[InputVariable]:
    prefix = (var.split("*", maxsplit=1)[0]).rstrip("-")
    match get_var_index([v.name for v in root.glob(var)], prefix):
        case Ok(idx):
            return Ok(InputVariable(prefix=prefix, root=root, idx=idx))
        case Err(e):
            return Err(e)


def parse_variable_input(var: str, root: Path) -> Result[InputVariable]:
    n_wildcard = var.count("*")
    match n_wildcard:
        case 0:
            return _parse_const_variable(var, root).next()
        case 1:
            return _parse_indexed_variable(var, root).next()
        case _:
            msg = f"Variable input {var} contains too many wildcards. Only one allowed."
            return Err(ValueError(msg))


def create_argument_list(
    var1: InputVariable, var2: InputVariable
) -> Generator[tuple[int | str, Path, Path]]:
    idx = var1.idx or var2.idx or ["N/A"]
    v_list1 = (
        VariableGeneratorConst(var1.prefix, var1.root, idx)
        if not var1.idx
        else VariableGenerator(var1.prefix, var1.root, idx)
    )
    v_list2 = (
        VariableGeneratorConst(var2.prefix, var2.root, idx)
        if not var2.idx
        else VariableGenerator(var2.prefix, var2.root, idx)
    )
    yield from zip(idx, v_list1, v_list2, strict=True)
