import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ._types import PrepArgs, PrepKwargs, SolverArgs, SolverKwargs, Verbosity

if TYPE_CHECKING:
    from collections.abc import Sequence


solver_parser = argparse.ArgumentParser("chsolve")
solver_parser.add_argument("pfile", nargs="+", type=Path)
solver_parser.add_argument("--cores", "-n", type=int, default=1)
solver_parser.add_argument("--log", action="store_true")
solver_parser.add_argument("--dump-matrix", action="store_true")
solver_parser.add_argument("--parallel", type=int, default=1)
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
    parallel: int = 1
    verbosity: Verbosity


class PrepModel(BaseModel):
    pfile: list[Path]
    log: bool = False
    verbosity: Verbosity


def parse_solver_cmdline_args(args: Sequence[str] | None = None) -> tuple[SolverArgs, SolverKwargs]:
    parsed_args = SolverModel(**vars(solver_parser.parse_args(args)))
    return (
        {"pfile": parsed_args.pfile, "parallel": parsed_args.parallel},
        SolverKwargs(
            cores=parsed_args.cores,
            log=parsed_args.log,
            dump_matrix=parsed_args.dump_matrix,
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
