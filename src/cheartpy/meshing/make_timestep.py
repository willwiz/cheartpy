#!/usr/bin/python
# -*- coding: utf-8 -*-

# Creates a time stepping scheme for the given input
# The input of the script are:
#     n1 size1 n2 size2 ... nn sizen fileout

import os
import sys
import fileinput
from cheartpy.tools.progress_bar import ProgressBar


arg = sys.argv

# Check for the number of arguments
#   The number of arguments must by at least 3, n, size, filename
if len(arg) < 4:
    print(
        "usage: \n\t The must have at least 3 arguments n, step size, filename \n\t./create_timestep.py n_1 size_1 ... n_n size_n filename_out"
    )
    exit()

if len(arg) % 2 != 0:
    print(
        "The number of arguments is incorrect. \n\t usage: \n\t./create_timestep.py n_1 size_1 ... n_n size_n filename_out"
    )
    exit()

# Get the filename to be saved to
fileout = arg[-1]
f = open(fileout, "w")

# Count the number of segments
n_seg = len(arg) / 2 - 1

# Convert the inputs to usable format
stepn = [int(i) for i in arg[1:-2:2]]
stepsize = [float(i) for i in arg[2::2]]

# Count the total number of time steps
n_total = sum(stepn)
# Write the total
f.write("{}".format(n_total))

# Print information on the inputs
print("There are {:d} segments:".format(n_seg))
for seg in range(n_seg):
    print(
        "\tsegment {}:\tn={}\tsize={}".format(
            seg + 1, arg[2 * seg + 1], arg[2 * seg + 2]
        )
    )

# Start the progress bar
bar = ProgressBar("Processing", max=n_total)

# Writing the time steps
i = 0  # This is the time step number
for n in range(n_seg):
    for j in range(stepn[n]):
        i = i + 1  # update the step number
        f.write("\n{} {}".format(i, stepsize[n]))
        bar.next()  # advance the progress bar
f.close()

print("\nJOB COMPLETE!!!")
