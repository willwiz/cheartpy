#!/usr/bin/python
# -*- coding: utf-8 -*-

# Takes a .T file for a line and produce a list of nodes on the line so that boundary conditions can be applied.


import sys
from cheartpy.tools.progress_bar import ProgressBar


arg = sys.argv

if len(arg) < 4:
    print("usage: \n\t./addconstraint.py Linefile.Pt val1 val2 ... valn fileout")
    exit()

filelin = arg[1]
fileout = "{}".format(arg[-1])
fout = open(fileout, "w")

# Get the number of constraints
ncons = len(arg) - 3

# First import the mesh data for the line
print("Processing the Ptfile data.")


with open(filelin, "r") as f:
    line = f.readline()
    fout.write(line)
    line = line.strip()
    items = line.split()
    b_elements = int(items[0])
    print("There are {} elements:".format(b_elements))
    bar = ProgressBar(b_elements, "Processing")
    for line in f:
        fout.write("{} ".format(line.strip()))
        for i in range(ncons):
            fout.write("{} ".format(arg[i + 2]))
        fout.write("\n")
        bar.next()
    bar.finish()

fout.close()
print("Operation complete!\n")
