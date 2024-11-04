import subprocess as sp


def run_prep(pfile: str):
    sp.run(["cheartsolver.out", pfile, "--prep"])


def run_problem(
    pfile: str,
    pedantic: bool = False,
    cores: int = 1,
    dump_matrix: bool = False,
    log: str | None = None,
):
    cmd = ["cheartsolver.out", pfile]
    if pedantic:
        cmd = cmd + ["--pedantic-printing"]
    if dump_matrix:
        cmd = cmd + ["--dump-matrix"]
    if cores > 1:
        cmd = ["mpiexec", "-n", f"{cores}"] + cmd
    print(" ".join(cmd))
    if log:
        with open(log, "w") as f:
            err = sp.check_call(cmd, stdout=f, stderr=sp.STDOUT)
    else:
        err = sp.check_call(cmd)
    print("cheartsolver.out has finished!")
    return err
