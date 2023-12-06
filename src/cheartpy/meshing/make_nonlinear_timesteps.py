#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Creates a time stepping scheme for a given minimum initial size, number of minimal steps and a maximum size
#  and creates a set of time steps with an exponential and smooth transition
# The inputs of this script are:
#     dt_start n_start n_trans n_total time_end fileout

import dataclasses as dc
from math import log
from typing import Literal
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from cheartpy.types import Arr, f64
import argparse


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
parser = argparse.ArgumentParser("maketime")
parser.add_argument("dt", type=float, help="initial dt")
parser.add_argument("n0", type=int, help="number of starting time steps with dt")
parser.add_argument("n1", type=int, help="number of time steps in the transition stage")
parser.add_argument("nt", type=int, help="total number of time steps")
parser.add_argument("tend", type=float, help="final time")
parser.add_argument("-p", "--prefix", type=str, default="time", help="prefix of output")


@dc.dataclass(slots=True)
class InputArgs:
    dt: float
    n0: int
    n1: int
    n2: int
    nt: int
    tend: float
    prefix: str


def check_args(nsp: argparse.Namespace) -> InputArgs:
    n2 = nsp.nt - nsp.n1 - nsp.n0
    if n2 < 1:
        raise ValueError("nt is less than n0 + n1")
    return InputArgs(nsp.dt, nsp.n0, nsp.n1, n2, nsp.nt, nsp.tend, nsp.prefix)


# These function compares two values and see if they are numerically equal given some numerical error from summing n floats
def float_equals(
    A: float, B: float, n: int, tol: float = 10.0
) -> Literal[True] | tuple[Literal[False], float]:
    trial: float = max(abs(A - np.full(n, A / n).sum()), np.finfo(float).eps)
    diff = abs(A - B)
    test = max(abs(A), abs(B))
    if diff <= tol * test * trial:
        return True
    print(f"{tol=}, {test=}, {trial=}")
    return False, tol * test * trial


def get_power_size(a: float, n: int) -> float:
    s: float  # sum
    b: float  # step size
    s, b = 0.0, a
    for _ in range(0, n):
        s, b = s + b, a * b
    return s


def get_power_max(a: float, n: int) -> float:
    b = a
    for _ in range(1, n):
        b = a * b
    return b


def compute_total_time(a: float, dt: float, n0: int, n1: int, n2: int):
    t0 = n0
    t1 = get_power_size(a, n1)
    t2 = get_power_max(a, n1) * n2
    return dt * (t0 + t1 + t2)


def find_parameter2(inp: InputArgs) -> float:
    def obf(x):
        y = compute_total_time(x[0], inp.dt, inp.n0, inp.n1, inp.n2)
        z = y - inp.tend
        return log(z * z)

    optres = minimize(
        obf, np.array([1.0]), bounds=Bounds(0.9, 1.001), method="TNC", tol=1e-14
    )
    return optres.x[0]


def mult_accumulate(n: int, a: float) -> Arr[int, f64]:
    res = np.full(n, a, dtype=float)
    return np.multiply.accumulate(res)


def create_dt(par: float, inp: InputArgs):
    dt = inp.dt * inp.tend / compute_total_time(par, inp.dt, inp.n0, inp.n1, inp.n2)
    dt0 = np.full(inp.n0, dt)
    dt1 = dt * mult_accumulate(inp.n1, par)
    dt2 = np.full(inp.n2, dt1[-1])
    print(
        f"The part 1 has {dt0.size} steps with size {dt0[0]} and {dt0.sum()} time elapsed"
    )
    print(
        f"The part 2 has {dt1.size} steps with multiplier {par} and {dt1.sum()} time elapsed"
    )
    print(
        f"The part 3 has {dt2.size} steps with size {dt2[0]} and {dt2.sum()} time elapsed"
    )
    return np.concatenate((dt0, dt1, dt2))


def export_dt(dt: Arr[int, f64], fout: str) -> None:
    with open(fout + ".step", "w") as f:
        f.write("{}".format(dt.size))
        for i, v in enumerate(dt, start=1):
            f.write("\n{} {}".format(i, v))
    list_t = np.add.accumulate(dt)
    with open(fout + ".tvi", "w") as f:
        f.write("{}".format(list_t.size))
        for i, v in enumerate(list_t, start=1):
            f.write("\n{} {}".format(i, v))
    return


def main(inp: InputArgs) -> None:
    par = find_parameter2(inp)
    dt = create_dt(par, inp)
    Tf_computed = dt.sum()
    match float_equals(Tf_computed, inp.tend, inp.nt):
        case True:
            pass
        case (False, v):
            raise ValueError(
                f"Total time {Tf_computed} is not the same as inputed {inp.tend} within tol = {v}. I did something wrong in the code!"
            )
    if not (dt.size == inp.nt):
        raise ValueError(
            f"The total number of time steps {dt.size} is not the same as requested {inp.nt}, I did something wrong appearantly."
        )
    print("Final Time Steps Generated:")
    print("    {} total time steps and {} time elapse".format(dt.size, dt.sum()))
    print("Now writing list of dt to file")
    export_dt(dt, inp.prefix)
    print("!!!JOB COMPLETE!!!")


def main_cli(args: argparse.Namespace) -> None:
    inp = check_args(args)
    main(inp)


if __name__ == "__main__":
    args = parser.parse_args()
    main_cli(args)
