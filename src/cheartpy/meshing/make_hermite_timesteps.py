#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Creates a time stepping scheme for a given minimum initial size, number of minimal steps and a maximum size
#  and creates a set of time steps with an exponential and smooth transition
# The inputs of this script are:
#     dt_start n_start n_trans n_total time_end fileout

import os
import sys
import fileinput

from math import exp
from math import log
from math import floor
from math import ceil
import numpy as np

# Get the precision for double below
dbl_eps = np.finfo(float).eps  # this really doens't work for python


# These function compares two values and see if they are numerically equal given some numerical error from summing n floats
def same_dbl(A, B, n):
    trial = abs(B - np.full(n, B / n).sum())
    diff = abs(A - B)
    test = max(abs(A), abs(B))
    if diff <= test * trial:
        return True
    return False


# This is the hermite polynomial with derivative of 0 at t = 0, & t = 1,
# initial valve of 0, and final value of 1
def hermite_continuous_function(t1):
    t2 = t1 * t1
    t3 = t2 * t1
    return -2.0 * t3 + 3.0 * t2


def get_hermite_array(n):
    mappedlist = [(float(i) + 0.5) / (float(n)) for i in range(n)]
    return np.asarray([hermite_continuous_function(t) for t in mappedlist])


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
arg = sys.argv

# Check the number of arguments to see if they are correct
#    The number of arguments must be exactly 6
if len(arg) != 7:
    print("{} arguments supplied".format(len(arg)))
    print("usage: \n\tThere must be exactly 6 arguments:")
    print("\tdt_start n_start n_trans n_total time_end fileout")
    exit()

# Get the filename and open a file
fileout_step = arg[-1] + ".step"
f_step = open(fileout_step, "w")
fileout_time = arg[-1] + ".tvi"
f_time = open(fileout_time, "w")

# Get the parameters from the inputs
dt0 = float(arg[1])
n0 = int(arg[2])
n1 = int(arg[3])
nt = int(arg[4])
n2 = nt - n1 - n0
if not (n2 > 0):
    raise ValueError(
        "Your total n is less than the number of step in the initial + transition range ... ..., please fix this ... ..."
    )
Tf = float(arg[5])

# Get the time steps in the initial range
list_dt0 = np.full(n0, dt0)
# Calculate the remaining time to be accounted for in the later two stages
t_baseline = nt * dt0
t_remaining = Tf - t_baseline
if not (t_remaining > 0.0):
    raise ValueError(
        "Your initial step size are too large, it already exceed the final time in {} steps".format(
            nt
        )
    )

# Compute the final dt
# Generate the unscaled transition time steps
list_dt1_unscaled = get_hermite_array(n1)
if not (list_dt1_unscaled.size == n1):
    raise ValueError(
        "For some reason the number of steps in the transition {} is not the same as requested {}".format(
            list_dt1_unscaled.size, n1
        )
    )
t_unscaled = list_dt1_unscaled.sum() + n2  # Compute the unscaled remaining time
dt2 = t_remaining / t_unscaled
if not (dt2 > 1e-12):
    raise ValueError(
        "Something went wrong! But your initial step size are too large!")
dtf = dt2 + dt0

# Compute the remaining list of dts and combine
list_dt1 = dt2 * list_dt1_unscaled + dt0
list_dt2 = np.full(n2, dtf)
list_dt = np.concatenate((list_dt0, list_dt1, list_dt2))
Tf_computed = list_dt.sum()


print("A list of dt has been generated with the following:")
print("    {} total time steps and {} time elapse".format(
    list_dt.size, list_dt.sum()))
print(
    "    The first  part has {} steps with size {} and {} time elapsed".format(
        list_dt0.size, list_dt0[0], list_dt0.sum()
    )
)
print(
    "    The second part has {} steps and {} time elapsed".format(
        list_dt1.size, list_dt1.sum()
    )
)
print(
    "    The third  part has {} steps with size {} and {} time elapsed".format(
        list_dt2.size, list_dt2[0], list_dt2.sum()
    )
)
print("Now writing list of dt to file")
if not same_dbl(Tf_computed, Tf, nt):
    raise ValueError(
        "Total time {} is not the same as inputed {}. I did something wrong in the code!".format(
            Tf_computed, Tf
        )
    )
if not (list_dt.size == nt):
    raise ValueError(
        "The total number of time steps is not the same as requested, I did something wrong appearantly."
    )

# Write the total number of time steps
f_step.write("{}".format(nt))
# Writing the time steps
for i in range(nt):
    f_step.write("\n{} {}".format(i + 1, list_dt[i]))
f_step.close

print("Now writing the list of time elasped to file")
list_t = np.add.accumulate(list_dt)
f_time.write("{}".format(nt))
# Writing the time steps
for i in range(nt):
    f_time.write("\n{} {}".format(i + 1, list_t[i]))
f_time.close
print("                !!!JOB COMPLETE!!!")
