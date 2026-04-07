from typing import TYPE_CHECKING, NamedTuple, TypedDict

if TYPE_CHECKING:
    from pathlib import Path


class Macros(NamedTuple):
    name: str
    value: str


class OutputPath(NamedTuple):
    path: Path


class ValueLine(NamedTuple):
    value: str


class Time(NamedTuple):
    name: str


class SolverGroup(NamedTuple):
    name: str


class SolverSubGroup(NamedTuple):
    name: str
    values: list[str]


class SolverGroupSetting(NamedTuple):
    name: str
    values: list[str]


class SolverMatrix(NamedTuple):
    name: str
    values: list[str]


class Basis(NamedTuple):
    name: str
    elem: str
    order: str
    gauss: str


class Topology(NamedTuple):
    name: str
    mesh: str
    basis: str


class Expression(NamedTuple):
    name: str


class Unparsed(NamedTuple):
    line: str


class Variable(NamedTuple):
    name: str
    top: str
    mesh: str | None
    dim: str


class Flag(NamedTuple):
    name: str


class Problem(NamedTuple):
    name: str
    type: str


type LineParseResult = (
    Macros
    | OutputPath
    | Expression
    | Time
    | SolverGroup
    | SolverSubGroup
    | SolverMatrix
    | Basis
    | Topology
    | Variable
    | Flag
    | ValueLine
    | Unparsed
    | None
)


class PFileObject(TypedDict, total=False):
    output_path: OutputPath
    time: dict[str, Time]
    macros: dict[str, str]
    lines: list[str]
