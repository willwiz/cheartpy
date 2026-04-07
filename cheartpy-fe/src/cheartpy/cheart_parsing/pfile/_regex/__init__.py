import re
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import (
    Basis,
    Expression,
    Flag,
    LineParseResult,
    Macros,
    OutputPath,
    Problem,
    SolverGroup,
    SolverGroupSetting,
    SolverSubGroup,
    Time,
    Topology,
    Unparsed,
    ValueLine,
    Variable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

MACRO = re.compile(r"#(?P<name>\w+)\s*=\s*(?P<value>.+)", re.IGNORECASE)
OUTPUT_PATH = re.compile(r"\s*!SetOutputPath=\{(?P<path>\w+)\}", re.IGNORECASE)
TIME = re.compile(r"\s*!DefTimeStepScheme\s*=\s*\{(?P<name>\w+)\}", re.IGNORECASE)
SOLVER_GROUP = re.compile(
    r"""\s*!DefSolverGroup\s*=\s*\{\s*
        (?P<name>\w+)\s*
        (?:\|\w+)*\}""",
    re.IGNORECASE | re.VERBOSE,
)
SOLVER_GROUP_SETTING = re.compile(
    r"""\s*!SetSolverGroup\s*=\s*\{\s*
        (?P<name>\w+)\s*\|\s*
        (?P<values>([\|\-\w]+))\}""",
    re.IGNORECASE | re.VERBOSE,
)
SOLVER_SUBGROUP = re.compile(
    r"""\s*!DefSolverSubGroup\s*=\s*\{\s*
        (?P<name>\w+)\s*\|\s*
        (?P<values>.+)\s*\}""",
    re.IGNORECASE | re.VERBOSE,
)
BASIS = re.compile(
    r"""\s*!UseBasis\s*=\s*\{\s*
        (?P<name>\w+)\s*\|\s*
        (?P<elem>\w+)\s*\|\s*
        (?P<order>\w+)\s*\|\s*
        (?P<gauss>\w+)\s*\}""",
    re.IGNORECASE | re.VERBOSE,
)
TOPOLOGY = re.compile(
    r"""\s*!DefTopology\s*=\s*\{\s*
        (?P<name>\w+)\s*\|\s*
        (?P<mesh>[\w\/]+)\s*\|\s*
        (?P<basis>\w+)\}""",
    re.IGNORECASE | re.VERBOSE,
)
EXPRESSION = re.compile(r"\s*!DefExpression\s*=\s*\{(?P<name>\w+)\}", re.IGNORECASE)
VARIABLE = re.compile(
    r"""!DefVariablePointer=\{\s*
        (?P<name>\w+)\s*\|\s*
        (?P<top>\w+)\s*\|\s*
        ((?P<mesh>.+)\s*\|\s*)?
        (?P<dim>\d+)\}""",
    re.IGNORECASE | re.VERBOSE,
)
FLAG = re.compile(r"\s*!(?P<name>[a-zA-Z]+(?:-[a-zA-Z]+))", re.IGNORECASE)
PROBLEM = re.compile(
    r"\s*!DefProblem\s*=\s*\{\s*(?P<name>\w+)\s*\|\s*(?P<type>\w+)\s*\}", re.IGNORECASE
)


def blank_line(line: str) -> bool:
    return line.startswith(r"%") or line == ""


def get_output_path(line: str) -> OutputPath | None:
    if match := OUTPUT_PATH.match(line):
        return OutputPath(Path(match.group("path")))
    return None


def get_macro(line: str) -> Macros | None:
    if match := MACRO.match(line):
        return Macros(match.group("name"), match.group("value"))
    return None


def get_time(line: str) -> Time | None:
    if match := TIME.match(line):
        return Time(match.group("name"))
    return None


def get_solver_group(line: str) -> SolverGroup | None:
    if match := SOLVER_GROUP.match(line):
        return SolverGroup(match.group("name"))
    return None


def get_solver_group_setting(line: str) -> SolverGroupSetting | None:
    if match := SOLVER_GROUP_SETTING.match(line):
        return SolverGroupSetting(
            match.group("name"), [v.strip() for v in match.group("values").split("|")]
        )
    return None


def get_solver_subgroup(line: str) -> SolverSubGroup | None:
    if match := SOLVER_SUBGROUP.match(line):
        return SolverSubGroup(
            name=match.group("name"), values=[v.strip() for v in match.group("values").split("|")]
        )
    return None


def get_problem(line: str) -> Problem | None:
    if match := PROBLEM.match(line):
        return Problem(name=match.group("name"), type=match.group("type"))
    return None


def get_variable(line: str) -> Variable | None:
    if match := VARIABLE.match(line):
        return Variable(
            match.group("name"), match.group("top"), match.group("mesh"), match.group("dim")
        )
    return None


def get_flag(line: str) -> Flag | None:
    if match := FLAG.match(line):
        return Flag(match.group("name"))
    return None


def get_expression(line: str) -> Expression | None:
    if match := EXPRESSION.match(line):
        return Expression(match.group("name"))
    return None


def get_basis(line: str) -> Basis | None:
    if match := BASIS.match(line):
        return Basis(
            name=match.group("name"),
            elem=match.group("elem"),
            order=match.group("order"),
            gauss=match.group("gauss"),
        )
    return None


def get_topology(line: str) -> Topology | None:
    if match := TOPOLOGY.match(line):
        return Topology(
            name=match.group("name"), mesh=match.group("mesh"), basis=match.group("basis")
        )
    return None


def get_unparsed(line: str) -> Unparsed | None:
    if line.startswith("!"):
        return Unparsed(line)
    return None


REGEX_LIST: list[Callable[[str], LineParseResult | None]] = [
    get_output_path,
    get_macro,
    get_time,
    get_solver_group,
    get_solver_group_setting,
    get_solver_subgroup,
    get_problem,
    get_expression,
    get_basis,
    get_topology,
    get_variable,
    get_flag,
    get_unparsed,
]


def parse_line(line: str) -> LineParseResult:
    # print(VARIABLE.match(r"!DefVariablePointer={F|TPQuad|9}"))
    if blank_line(line.strip()):
        return None
    for regex in REGEX_LIST:
        match regex(line.strip()):
            case None:
                continue
            case result:
                return result
    return ValueLine(line.strip())
