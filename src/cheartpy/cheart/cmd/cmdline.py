from typing import TypedDict, Unpack

__all__ = ["run_prep", "run_problem"]
import subprocess as sp
from pathlib import Path


class _RunOptions(TypedDict, total=False):
    pedantic: bool
    dump_matrix: bool
    output: bool
    cores: int
    log: str | None


def run_prep(pfile: str) -> None:
    sp.run(["cheartsolver.out", pfile, "--prep"], check=False)


def run_problem(pfile: str, **kwargs: Unpack[_RunOptions]) -> int:
    cmd = ["cheartsolver.out", pfile]
    cores = kwargs.get("cores", 1)
    if cores > 1:
        cmd = ["mpiexec", "-n", f"{cores}", *cmd]
    if kwargs.get("pedantic", False):
        cmd = [*cmd, "--pedantic-printing"]
    if kwargs.get("dump_matrix", False):
        cmd = [*cmd, "--dump-matrix"]
    if not kwargs.get("output", False):
        cmd = [*cmd, "--no-output"]
    print(" ".join(cmd))
    log = kwargs.get("log")
    if log:
        with Path(log).open("w") as f:
            err = sp.check_call(cmd, stdout=f, stderr=sp.STDOUT)
    else:
        err = sp.check_call(cmd)
    print("cheartsolver.out has finished!")
    return err
