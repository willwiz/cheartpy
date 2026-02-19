import subprocess as sp
from pathlib import Path
from typing import TypedDict, Unpack

from pytools.logging import get_logger


class _RunOptions(TypedDict, total=False):
    pedantic: bool
    dump_matrix: bool
    output: bool
    cores: int
    log: Path | str


def run_prep(pfile: Path | str, **kwargs: Unpack[_RunOptions]) -> int:
    cmd = ["cheartsolver.out", str(pfile), "--prep"]
    if kwargs.get("output", True) is False:
        cmd = [*cmd, "--no-output"]
    logger = get_logger()
    logger.disp(" ".join(cmd))
    if (log := kwargs.get("log")) is not None:
        with Path(log).open("w") as f:
            err = sp.run(cmd, stdout=f, stderr=sp.STDOUT, check=False).returncode
    else:
        err = sp.run(cmd, check=False).returncode
    return err


def run_problem(pfile: Path | str, **kwargs: Unpack[_RunOptions]) -> int:
    cmd = ["cheartsolver.out", str(pfile)]
    cores = kwargs.get("cores", 1)
    if cores > 1:
        cmd = ["mpiexec", "-n", f"{cores}", *cmd]
    if kwargs.get("pedantic", False):
        cmd = [*cmd, "--pedantic-printing"]
    if kwargs.get("dump_matrix", False):
        cmd = [*cmd, "--dump-matrix"]
    if kwargs.get("output", True) is False:
        cmd = [*cmd, "--no-output"]
    logger = get_logger()
    logger.disp(" ".join(cmd))

    if (log := kwargs.get("log")) is not None:
        with Path(log).open("w") as f:
            err = sp.run(cmd, stdout=f, stderr=sp.STDOUT, check=False).returncode
    else:
        err = sp.run(cmd, check=False).returncode
    logger.disp(f"{pfile!s} has finished!")
    return err
