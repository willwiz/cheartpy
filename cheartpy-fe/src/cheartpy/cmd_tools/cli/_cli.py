import subprocess as sp
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from cheartpy.cheart_parsing.pfile.find import find_output_dir
from pytools.logging import get_logger

from ._parser import parse_prep_cmdline_args, parse_solver_cmdline_args
from ._types import CheartErrorCode, PrepKwargs, SolverKwargs, Verbosity

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_prep(pfile: Path | str, **kwargs: Unpack[PrepKwargs]) -> int:
    pfile = Path(pfile)
    cmd = ["cheartsolver.out", str(pfile), "--prep"]
    match kwargs.get("verbosity", Verbosity.NONE):
        case Verbosity.PEDANTIC:
            cmd = [*cmd, "--pedantic-printing"]
        case Verbosity.QUIET:
            cmd = [*cmd, "--no-output"]
        case Verbosity.NONE:
            pass
    logger = kwargs.get("logger", get_logger())
    logger.disp(" ".join(cmd))
    if kwargs.get("log"):
        with pfile.with_suffix(".prep").open("w") as f:
            return sp.run(cmd, stdout=f, stderr=f, check=False).returncode
    return sp.run(cmd, check=False).returncode


def run_problem(pfile: Path | str, **kwargs: Unpack[SolverKwargs]) -> int:
    pfile = Path(pfile)
    output_path = find_output_dir(pfile)
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
    cmd = ["cheartsolver.out", str(pfile)]
    cores = kwargs.get("cores", 1)
    if cores > 1:
        cmd = ["mpiexec", "-n", f"{cores}", *cmd]
    match kwargs.get("verbosity", Verbosity.NONE):
        case Verbosity.PEDANTIC:
            cmd = [*cmd, "--pedantic-printing"]
        case Verbosity.QUIET:
            cmd = [*cmd, "--no-output"]
        case Verbosity.NONE:
            pass
    if kwargs.get("dump_matrix", False):
        cmd = [*cmd, "--dump-matrix"]
    logger = kwargs.get("logger", get_logger())
    logger.disp(" ".join(cmd))
    if kwargs.get("log"):
        with (
            pfile.with_suffix(".log").open("w") as f,
            pfile.with_suffix(".err").open("w") as ferr,
        ):
            return sp.run(cmd, stdout=f, stderr=ferr, check=False).returncode
    return sp.run(cmd, check=False).returncode


def solver_cli(args: Sequence[str] | None = None) -> None:
    _args, _kwargs = parse_solver_cmdline_args(args)
    logger = get_logger()
    errs = []
    for p in _args["pfile"]:
        errs.append(run_problem(p, logger=logger, **_kwargs))
        if errs[-1] == CheartErrorCode.SUCCESS.value:
            logger.disp(f"Problem {p} solved successfully.")
        else:
            err_code = (
                f"{CheartErrorCode(errs[-1]).name!s}"
                if errs[-1] in CheartErrorCode._value2member_map_
                else f"{CheartErrorCode.UNKNOWN.name!s} = {errs[-1]}"
            )
            logger.err(f"Problem {p} failed with error code {err_code}.")
    if any(errs):
        msg = f"One or more problems failed with errors: {errs}"
        raise RuntimeError(msg)


def prep_cli(args: Sequence[str] | None = None) -> None:
    _args, _kwargs = parse_prep_cmdline_args(args)
    logger = get_logger()
    errs = []
    for p in _args["pfile"]:
        errs.append(run_prep(p, **_kwargs))
        if errs[-1] == CheartErrorCode.SUCCESS.value:
            logger.disp(f"Prep for {p} completed successfully.")
        else:
            err_code = (
                f"{CheartErrorCode(errs[-1]).name!s}"
                if errs[-1] in CheartErrorCode._value2member_map_
                else f"{CheartErrorCode.UNKNOWN.name!s} = {errs[-1]}"
            )
            logger.err(f"Prep for {p} failed with error code {err_code}.")
    if any(errs):
        msg = f"One or more prep tasks failed with errors: {errs}"
        raise RuntimeError(msg)
