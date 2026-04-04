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
solver_parser.set_defaults(verbosity=Verbosity.NONE)
verbosity = solver_parser.add_mutually_exclusive_group()
verbosity.add_argument(
    "--verbose", dest="verbosity", action="store_const", const=Verbosity.PEDANTIC
)
verbosity.add_argument("--quiet", dest="verbosity", action="store_const", const=Verbosity.QUIET)


prep_parser = argparse.ArgumentParser("chprep")
prep_parser.add_argument("pfile", nargs="+", type=Path)
prep_parser.add_argument("--log", action="store_true")


class SolverModel(BaseModel):
    pfile: list[Path]
    cores: int = 1
    log: bool = False
    dump_matrix: bool = False
    verbose: bool = False
    quiet: bool = False


class PrepModel(BaseModel):
    pfile: list[Path]
    log: bool = False


def parse_solver_cmdline_args(args: Sequence[str] | None = None) -> tuple[SolverArgs, SolverKwargs]:
    parsed_args = SolverModel(**vars(solver_parser.parse_args(args)))
    return (
        {"pfile": parsed_args.pfile},
        {
            "cores": parsed_args.cores,
            "log": parsed_args.log,
            "dump_matrix": parsed_args.dump_matrix,
            "verbose": parsed_args.verbose,
            "quiet": parsed_args.quiet,
        },
    )


def parse_prep_cmdline_args(args: Sequence[str] | None = None) -> tuple[PrepArgs, PrepKwargs]:
    parsed_args = PrepModel(**vars(prep_parser.parse_args(args)))
    return (
        {"pfile": parsed_args.pfile},
        {"log": parsed_args.log},
    )
