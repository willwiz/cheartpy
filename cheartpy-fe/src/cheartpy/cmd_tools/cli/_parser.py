import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pytools.logging import BColors
from pytools.result import Err, Ok, Result

from ._types import PrepArgs, PrepKwargs, SolverArgs, SolverKwargs, Verbosity

if TYPE_CHECKING:
    from collections.abc import Sequence


solver_parser = argparse.ArgumentParser("chsolve")
solver_parser.add_argument("pfile", nargs="+", type=Path)
solver_parser.add_argument("--cores", "-n", type=int, default=1, metavar="MPI_CORES(int)")
solver_parser.add_argument("--log", action="store_true")
solver_parser.add_argument("--dump-matrix", action="store_true")
solver_parser.add_argument("--dump-residual", action="store_true")
solver_parser.add_argument("--parallel", type=int, default=1, metavar="PARALLEL_RUNS(int)")
solver_parser.add_argument(
    "--macro",
    "-m",
    "-#",
    action="append",
    type=str,
    metavar="KEY(str)=VALUE(str)",
    help="REPEATABLE",
)
solver_parser.set_defaults(verbosity="DEFAULT")
verbosity = solver_parser.add_mutually_exclusive_group()
verbosity.add_argument(
    "--verbose",
    "-v",
    dest="verbosity",
    action="store_const",
    const="PEDANTIC",
)
verbosity.add_argument(
    "--quiet",
    "-q",
    dest="verbosity",
    action="store_const",
    const="QUIET",
)


prep_parser = argparse.ArgumentParser("chprep")
prep_parser.add_argument("pfile", nargs="+", type=Path)
prep_parser.add_argument("--log", action="store_true")
prep_parser.add_argument(
    "--quiet", dest="verbosity", action="store_const", const="QUIET", default="DEFAULT"
)


class SolverModel(BaseModel):
    pfile: list[Path]
    cores: int = 1
    log: bool = False
    dump_matrix: bool = False
    dump_residual: bool = False
    macro: list[str] | None = None
    parallel: int = 1
    verbosity: Verbosity


class PrepModel(BaseModel):
    pfile: list[Path]
    log: bool = False
    verbosity: Verbosity

RX_MACRO = re.compile(r"\s*(?P<key>.+)=(?P<value>.+)\s*")


def parse_macro(*macro: str) -> Result[dict[str, str]]:
    if not macro:
        return Ok({})
    macros = {m: RX_MACRO.fullmatch(m) for m in macro}
    false_macros = {k: v for k, v in macros.items() if v is None}
    if false_macros:
        msg = f"Invalid macro definitions (--macro [-m] k=v):\n{'\n'.join(false_macros.keys())}"
        return Err(ValueError(msg))
    return Ok({v.group("key"): v.group("value") for v in macros.values() if v is not None})


def parse_solver_cmdline_args(args: Sequence[str] | None = None) -> tuple[SolverArgs, SolverKwargs]:
    parsed_args = SolverModel(**vars(solver_parser.parse_args(args)))
    match parse_macro(*(parsed_args.macro or [])):
        case Ok(macros): ...  # fmt: skip
        case Err(e):
            solver_parser.print_help()
            print(f"\n{BColors.FAIL}{e!s}{BColors.ENDC}")
            raise SystemExit(1)
    return (
        {"pfile": parsed_args.pfile, "parallel": parsed_args.parallel},
        SolverKwargs(
            cores=parsed_args.cores,
            log=parsed_args.log,
            dump_matrix=parsed_args.dump_matrix,
            dump_residual=parsed_args.dump_residual,
            macros=macros,
            verbosity=parsed_args.verbosity,
        ),
    )


def parse_prep_cmdline_args(args: Sequence[str] | None = None) -> tuple[PrepArgs, PrepKwargs]:
    parsed_args = PrepModel(**vars(prep_parser.parse_args(args)))
    return (
        {"pfile": parsed_args.pfile},
        PrepKwargs(
            log=parsed_args.log,
            verbosity=parsed_args.verbosity,
        ),
    )
