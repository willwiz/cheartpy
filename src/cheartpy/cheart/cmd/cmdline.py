from __future__ import annotations

__all__ = ["run_prep", "run_problem"]
import subprocess as sp
from pathlib import Path


def run_prep(pfile: str) -> None:
    sp.run(["cheartsolver.out", pfile, "--prep"], check=False)


def run_problem(
    pfile: str,
    *,
    pedantic: bool = False,
    dump_matrix: bool = False,
    output: bool = True,
    cores: int = 1,
    log: str | None = None,
) -> int:
    cmd = ["cheartsolver.out", pfile]
    if pedantic:
        cmd = [*cmd, "--pedantic-printing"]
    if dump_matrix:
        cmd = [*cmd, "--dump-matrix"]
    if cores > 1:
        cmd = ["mpiexec", "-n", f"{cores}", *cmd]
    if not output:
        cmd = [*cmd, "--no-output"]
    print(" ".join(cmd))
    if log:
        with Path(log).open("w") as f:
            err = sp.check_call(cmd, stdout=f, stderr=sp.STDOUT)
    else:
        err = sp.check_call(cmd)
    print("cheartsolver.out has finished!")
    return err
