#!/usr/bin/python
# -*- coding: utf-8 -*-

# find the nodes in a file.X that satisfies the constraints given
# The inputs of this script are:
#     filein cons1 cons2 ... consn fileout

import sys
from cheartpy.tools.progress_bar import progress_bar

# Get the precision for double below
dbl_eps = 4.4408920985e-15


# These function compares two values and see if they are numerically equal
def same_dbl(A):
    if abs(A) < dbl_eps:
        return True
    return False


# Get the command line arguments
arg = sys.argv

# Get the number of constraints
ncons = len(arg) - 3

# Check the number of arguments
# The number of arguments must be 3 or more
if len(arg) < 1:
    print("{} arguments supplied.".format(len(arg) - 1))
    print("There must be at least 3 arguments")
    print("usage:")
    print("\tfilein cons1 cons2 ... consn fileout")
    exit()

# Get the filename and open a file
filein = arg[1]
fin = open(filein, "r")

# Get the filename and open a file
fileout = arg[-1]
fout = open(fileout, "w")

# Read in the constraints
with open(filein, "r") as f:
    line = f.readline()
    line = line.strip()
    items = line.split()
    nodes_total = int(items[0])
    print("There are {} points in the X file".format(nodes_total))
    collect = []
    bar = progress_bar("Importing data", max=nodes_total)
    try:
        for line in f:
            line = line.strip()
            items = [float(i) for i in line.split()]
            collect.append(items)
            bar.next()
    except:
        print()
    bar.finish()
n_points = len(collect)

# Check to see if the X file has errors
if nodes_total != n_points:
    print(
        "The number of nodes indicated in the header of the X file {} \
        is not equal to the number of points following the header {}".format(
            nodes_total, n_points
        )
    )
    exit()

# Generate a list of points cooresponding to the nodes
points = range(n_points)

# Start eliminating the points based on the constraints in order
for i in range(ncons):
    # Set up a work array
    func = lambda x, y, z: eval(str(arg[i + 2]))
    if isinstance(func(0.0, 0.0, 0.0), (bool)):
        points = [
            p for p in points if func(collect[p][0], collect[p][1], collect[p][2])
        ]
    elif isinstance(func(0.0, 0.0, 0.0), (int, float)):
        points = [
            p
            for p in points
            if same_dbl(func(collect[p][0], collect[p][1], collect[p][2]))
        ]
    else:
        print("The {}th constraint given cannot be evaluated!!!".format(i))
        print("Please make sure the inequalities are functions of x, y, z")
        exit()
    n_points = len(points)
    print("Array has been filtered by the {}th constraint".format(i))

print("\nNow writing to file.")
fout.write("{}".format(n_points))
for i in range(n_points):
    fout.write("\n{}".format(points[i] + 1))
print("The filter array has been written to {}".format(fileout))
