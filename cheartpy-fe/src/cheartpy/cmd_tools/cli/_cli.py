import subprocess as sp
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from cheartpy.cheart_parsing.pfile.find import find_output_dir
from pytools.logging import get_logger
from pytools.parallel import ThreadedRunner

from ._parser import parse_prep_cmdline_args, parse_solver_cmdline_args
from ._types import CheartErrorCode, PrepKwargs, SolverKwargs

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_prep(pfile: Path | str, **kwargs: Unpack[PrepKwargs]) -> int:
    pfile = Path(pfile)
    cmd = ["cheartsolver.out", str(pfile), "--prep"]
    match kwargs.get("verbosity") or "DEFAULT":
        case "PEDANTIC" | "DEFAULT" | "NONE":
            pass
        case "QUIET":
            cmd = [*cmd, "--no-output"]
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
    if (cores := kwargs.get("cores", 1)) > 1:
        cmd = ["mpiexec", "-n", f"{cores}", *cmd]
    match kwargs.get("verbosity") or "DEFAULT":
        case "PEDANTIC":
            cmd = [*cmd, "--pedantic-printing"]
        case "QUIET":
            cmd = [*cmd, "--no-output"]
        case "NONE" | "DEFAULT":
            pass
    if kwargs.get("dump_matrix", False):
        cmd = [*cmd, "--dump-matrix"]
    if kwargs.get("dump_residual", False):
        cmd = [*cmd, "--dump-residual"]
    if macros := kwargs.get("macros"):
        cmd = [*cmd, *(f"-#{k}={v}" for k, v in macros.items())]
    logger = kwargs.get("logger", get_logger())
    logger.disp(" ".join(cmd))
    if kwargs.get("log"):
        with (
            pfile.with_suffix(".log").open("w") as f,
            pfile.with_suffix(".err").open("w") as ferr,
        ):
            return sp.run(cmd, stdout=f, stderr=ferr, check=False).returncode
    return sp.run(cmd, check=False).returncode


def solver_cli_series(*pfiles: Path | str, **kwargs: Unpack[SolverKwargs]) -> None:
    logger = kwargs.get("logger") or get_logger()
    errs: list[int] = []
    for p in pfiles:
        errs.append(run_problem(p, **kwargs))
        if errs[-1] == CheartErrorCode.SUCCESS.value:
            logger.disp(f"Problem {p} solved successfully.")
        else:
            err_code = (
                f"{CheartErrorCode(errs[-1]).name!s}"
                if errs[-1] in CheartErrorCode._value2member_map_
                else f"{CheartErrorCode.UNKNOWN.name!s} = {errs[-1]}"
            )
            logger.error(f"Problem {p} failed with error code {err_code}.")
    if any(errs):
        msg = f"One or more problems failed with errors: {errs}"
        raise RuntimeError(msg)


def solver_cli_parallel(
    *pfiles: Path | str, nthreads: int = 1, **kwargs: Unpack[SolverKwargs]
) -> None:
    with ThreadedRunner(thread=nthreads) as runner:
        [runner.submit(run_problem, p, **kwargs) for p in pfiles]


def solver_cli(args: Sequence[str] | None = None) -> None:
    _args, _kwargs = parse_solver_cmdline_args(args)
    logger = get_logger()
    _kwargs: SolverKwargs = {"logger": logger, **_kwargs}
    if _args["parallel"] > 1:
        solver_cli_parallel(*_args["pfile"], nthreads=_args["parallel"], **_kwargs)
    else:
        solver_cli_series(*_args["pfile"], **_kwargs)


def prep_cli(args: Sequence[str] | None = None) -> None:
    _args, _kwargs = parse_prep_cmdline_args(args)
    logger = get_logger()
    errs: list[int] = []
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
            logger.error(f"Prep for {p} failed with error code {err_code}.")
    if any(errs):
        msg = f"One or more prep tasks failed with errors: {errs}"
        raise RuntimeError(msg)
