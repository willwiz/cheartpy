import subprocess as sp
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

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
    logger = get_logger()
    logger.disp(" ".join(cmd))
    if kwargs.get("log"):
        with pfile.with_suffix(".prep").open("w") as f:
            return sp.run(cmd, stdout=f, stderr=f, check=False).returncode
    return sp.run(cmd, check=False).returncode


def run_problem(pfile: Path | str, **kwargs: Unpack[SolverKwargs]) -> int:
    pfile = Path(pfile)
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
    logger = get_logger()
    logger.disp(" ".join(cmd))

    if kwargs.get("log"):
        with (
            pfile.with_suffix(".log").open("w") as f,
            pfile.with_suffix(".err").open("w") as ferr,
        ):
            err = sp.run(cmd, stdout=f, stderr=ferr, check=False).returncode
    else:
        err = sp.run(cmd, check=False).returncode
    if err == 0:
        logger.disp(f"{pfile!s} has finished!")
    else:
        err_code = (
            f"{CheartErrorCode(err).name}"
            if err in CheartErrorCode._value2member_map_
            else f"{CheartErrorCode.UNKNOWN} = {err}"
        )
        logger.error(f"{pfile!s} has failed with error code {err_code}")
    return err


def solver_cli(args: Sequence[str] | None = None) -> None:
    _args, _kwargs = parse_solver_cmdline_args(args)
    errs = [run_problem(p, **_kwargs) for p in _args["pfile"]]
    if any(errs):
        msg = f"One or more problems failed with errors: {errs}"
        raise RuntimeError(msg)


def prep_cli(args: Sequence[str] | None = None) -> None:
    _args, _kwargs = parse_prep_cmdline_args(args)
    errs = [run_prep(p, **_kwargs) for p in _args["pfile"]]
    if any(errs):
        msg = f"One or more prep tasks failed with errors: {errs}"
        raise RuntimeError(msg)
