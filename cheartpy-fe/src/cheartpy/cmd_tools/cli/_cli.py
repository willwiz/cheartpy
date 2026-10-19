import enum
import shutil
import subprocess as sp
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Unpack

from cheartpy.cheart_parsing.pfile.find import find_output_dir
from pytools.logging import get_logger
from pytools.parallel import ThreadedRunner

from ._parser import parse_prep_cmdline_args, parse_solver_cmdline_args
from ._types import CheartErrorCode, PrepKwargs, SolverKwargs

if TYPE_CHECKING:
    from collections.abc import Sequence


class MPIImplementation(enum.Enum):
    none = enum.auto()
    mpich = enum.auto()
    openmpi = enum.auto()


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


def _check_mpiexec_version() -> Literal["mpich", "openmpi", "none"]:
    mpiexec_path = shutil.which("mpiexec")
    if not mpiexec_path:
        return "none"
    result = ";".join(
        [
            r.stdout.decode("utf-8", errors="ignore").lower()
            for v in ["--version", "-version"]
            if ((r := sp.run(["mpiexec", v], capture_output=True, check=False)).returncode == 0)
        ]
    )
    if "open mpi" in result or "open-mpi" in result:
        return "openmpi"

    if "mpich" in result or "hydra build" in result:
        return "mpich"

    return "none"


def _parse_mpi_keys(**kwargs: object) -> list[str]:
    match kwargs.get("opt", _check_mpiexec_version()):
        case "mpich":
            return ["-bind-to", "core", "-map-by", "core"]
        case "openmpi":
            return ["--bind-to", "core", "--map-by", "core"]
        case _:
            return []


def run_problem(pfile: Path | str, **kwargs: Unpack[SolverKwargs]) -> int:
    pfile = Path(pfile)
    output_path = find_output_dir(pfile)
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
    if kwargs.get("clear") and output_path:
        for f in Path(output_path).glob("*"):
            f.unlink()
    cmd = ["cheartsolver.out", str(pfile)]
    mpi_keys = _parse_mpi_keys(**kwargs)
    if (cores := kwargs.get("cores", 1)) > 1:
        cmd = ["mpiexec", "-n", f"{cores}", *mpi_keys, *cmd]
    match kwargs.get("verbosity") or "DEFAULT":
        case "PEDANTIC":
            cmd = [*cmd, "--pedantic-printing"]
        case "QUIET":
            cmd = [*cmd, "--no-output"]
        case "NONE" | "DEFAULT":
            pass
    extra_flags = [
        f"--{k.replace('_', '-')}"
        for k in ["dump_matrix", "dump_residual", "dump_intermediate"]
        if kwargs.get(k, False)
    ]
    cmd = [*cmd, *extra_flags]
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
    if len(_args["pfile"]) > 1:
        logger.info(
            "Queuing problems for parallel execution...",
            nthreads=_args["parallel"],
            pfiles=_args["pfile"],
        )
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
