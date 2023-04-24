#!/usr/bin/python3
# -*- coding: utf-8 -*-

# This creates an hexahedral mesh of uniform grids
# The inputs of this script are:
#     x_length x_n x_offset y_length y_n y_offset z_length z_n z_offset fileout
# This 3 files created are:
#     Topology.T
#        n dim
#        node1 node2 ...
#        .
#        .
#
#     Node.X
#        n dim
#        x1 x2 x3 ...
#        .
#        .
#
#     Boundary.B
#        n
#        elem node1 node2 ... patch#
#        .
#        .
#

import os, sys, fileinput
import numpy as np

# This is gives the order of the nodes in an elent given a starting indice
vorder = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

# These function compares two values and see if they are numerically equal given some numerical error from summing n floats
def same_dbl(A, B, n):
  trial = abs(A - np.full(n, A/n).sum())
  diff = abs(A - B)
  test = max(abs(A), abs(B))
  if (diff <= test * trial):
    return True
  return False

class make_grid:
    tp   = []
    tpi  = []
    surf = []
    edge = []
    def __init__(self, arg):
        # extract the arguments first
        self.xsize = float(arg[1])
        self.xn    = int(arg[2])
        self.dx    = float(arg[1])/float(arg[2])
        self.xoff  = float(arg[3])
        self.ysize = float(arg[4])
        self.yn    = int(arg[5])
        self.dy    = float(arg[4])/float(arg[5])
        self.yoff  = float(arg[6])
        self.zsize = float(arg[7])
        self.zn    = int(arg[8])
        self.dz    = float(arg[7])/float(arg[8])
        self.zoff  = float(arg[9])
        #
        self.nnode = (int(arg[2]) + 1) * (int(arg[5]) + 1) * (int(arg[8]) + 1)
        self.nelem = int(arg[2]) * int(arg[5]) * int(arg[8])
        # create empty index array
        self.ind = np.zeros((self.xn + 1, self.yn + 1, self.zn +1), dtype=int)
        # fill array
        g = 0
        for i in range(self.zn + 1):
            for j in range(self.yn + 1):
                for k in range(self.xn + 1):
                    self.ind[k, j, i] = g
                    g = g + 1
        size = (self.xn + 1) * (self.yn + 1) * (self.zn + 1)
        # create empty space array
        self.space = np.zeros((size, 3))
        # fill array
        for i in range(self.zn + 1):
            for j in range(self.yn + 1):
                for k in range(self.xn + 1):
                    self.space[self.ind[k, j, i]] = [k*self.dx + self.xoff, j*self.dy + self.yoff, i*self.dz + self.zoff]
    def __getitem__(self, key):
        # if key is of invalid type or value, the list values will raise the error
        return self.space[self.ind[key]]

def compute_topology(g):
    g.tp = np.zeros((g.nelem, 8), dtype=int)
    for i in range(g.zn):
        for j in range(g.yn):
            for k in range(g.xn):
                for m in range(8):
                    g.tp[i * g.yn * g.xn + j * g.xn + k, m] = g.ind[k + vorder[m][0], j + vorder[m][1], i + vorder[m][2]]
    g.tpi = np.zeros((g.xn, g.yn, g.zn), dtype=int)
    l = 0
    for i in range(g.zn):
        for j in range(g.yn):
            for k in range(g.xn):
                g.tpi[k, j, i] = l
                l = l + 1
    return

def compute_surface(g):
    # for the -x surface
    surface = np.zeros((g.yn * g.zn, 6), dtype=int)
    for i in range(g.zn):
        for j in range(g.yn):
            surface[i*g.yn + j, 0] = g.tpi[0, j,     i]
            surface[i*g.yn + j, 1] = g.ind[0, j,     i]
            surface[i*g.yn + j, 2] = g.ind[0, j + 1, i]
            surface[i*g.yn + j, 3] = g.ind[0, j,     i + 1]
            surface[i*g.yn + j, 4] = g.ind[0, j + 1, i + 1]
            surface[i*g.yn + j, 5] = 0
    g.surf.append(surface)
    # for the +x surface
    surface = np.zeros((g.yn * g.zn, 6), dtype=int)
    for i in range(g.zn):
        for j in range(g.yn):
            surface[i*g.yn + j, 0] = g.tpi[g.xn - 1, j,     i]
            surface[i*g.yn + j, 1] = g.ind[g.xn, j,     i]
            surface[i*g.yn + j, 2] = g.ind[g.xn, j + 1, i]
            surface[i*g.yn + j, 3] = g.ind[g.xn, j,     i + 1]
            surface[i*g.yn + j, 4] = g.ind[g.xn, j + 1, i + 1]
            surface[i*g.yn + j, 5] = 1
    g.surf.append(surface)
    # for the -y surface
    surface = np.zeros((g.xn * g.zn, 6), dtype=int)
    for i in range(g.zn):
        for k in range(g.xn):
            surface[i*g.xn + k, 0] = g.tpi[k,     0, i]
            surface[i*g.xn + k, 1] = g.ind[k,     0, i]
            surface[i*g.xn + k, 2] = g.ind[k + 1, 0, i]
            surface[i*g.xn + k, 3] = g.ind[k,     0, i + 1]
            surface[i*g.xn + k, 4] = g.ind[k + 1, 0, i + 1]
            surface[i*g.xn + k, 5] = 2
    g.surf.append(surface)
    # for the +y surface
    surface = np.zeros((g.xn * g.zn, 6), dtype=int)
    for i in range(g.zn):
        for k in range(g.xn):
            surface[i*g.xn + k, 0] = g.tpi[k,     g.yn - 1 , i]
            surface[i*g.xn + k, 1] = g.ind[k,     g.yn, i]
            surface[i*g.xn + k, 2] = g.ind[k + 1, g.yn, i]
            surface[i*g.xn + k, 3] = g.ind[k,     g.yn, i + 1]
            surface[i*g.xn + k, 4] = g.ind[k + 1, g.yn, i + 1]
            surface[i*g.xn + k, 5] = 3
    g.surf.append(surface)
    # for the -z surface
    surface = np.zeros((g.xn * g.yn, 6), dtype=int)
    for j in range(g.yn):
        for k in range(g.xn):
            surface[j*g.xn + k, 0] = g.tpi[k,     j,     0]
            surface[j*g.xn + k, 1] = g.ind[k,     j,     0]
            surface[j*g.xn + k, 2] = g.ind[k + 1, j,     0]
            surface[j*g.xn + k, 3] = g.ind[k,     j + 1, 0]
            surface[j*g.xn + k, 4] = g.ind[k + 1, j + 1, 0]
            surface[j*g.xn + k, 5] = 4
    g.surf.append(surface)
    # for the +z surface
    surface = np.zeros((g.xn * g.yn, 6), dtype=int)
    for j in range(g.yn):
        for k in range(g.xn):
            surface[j*g.xn + k, 0] = g.tpi[k,     j,     g.zn - 1]
            surface[j*g.xn + k, 1] = g.ind[k,     j,     g.zn]
            surface[j*g.xn + k, 2] = g.ind[k + 1, j,     g.zn]
            surface[j*g.xn + k, 3] = g.ind[k,     j + 1, g.zn]
            surface[j*g.xn + k, 4] = g.ind[k + 1, j + 1, g.zn]
            surface[j*g.xn + k, 5] = 5
    g.surf.append(surface)
    return

def compute_edge(g):
    g.tp = np.zeros((g.zn * g.yn * g.xn, 8), dtype=int)
    for i in range(g.zn):
        for j in range(g.yn):
            for k in range(g.xn):
                for m in range(8):
                    g.tp[i * g.yn * g.xn + j * g.xn + k, m] = g.ind[i + vorder[m][0], j + vorder[m][1], k + vorder[m][2]]
    return

# g.ind[[i + order[m, 0], j + order[m, 1], k + order[m, 2]]]





# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
arg = sys.argv

# Check the number of arguments to see if they are correct
#    The number of arguments must be exactly 6
if len(arg) != 11:
  print('{} arguments supplied'.format(len(arg)))
  print('usage: \n\tThere must be exactly 10 arguments:')
  print('\tx_length x_n x_offset y_length y_n y_offset z_length z_n z_offset fileout')
  exit()

# Create the filenames for export
fileX = arg[-1] + '_FE.X'
fileT = arg[-1] + '_FE.T'
fileB = arg[-1] + '_FE.B'


g = make_grid(arg)
print('Made a grid with {} nodes'.format(g.nnode))
compute_topology(g)
print('Created a topology for the grid with {} elements'.format(g.nelem))
compute_surface(g)
print('Generated a the 6 surfaces facing each side')

print(g.space)
print(g.tp)

with open(fileX, 'w') as f:
    f.write('{:>12d}{:>12d}'.format(g.nnode, 3))
    for [x, y, z] in g.space:
        print(x, y, z)
        f.write('\n{:>20.16}{:>20.16f}{:>20.16f}'.format(x, y, z))
print('The X file is saved as {}'.format(fileX))

with open(fileT, 'w') as f:
    f.write('{:>12d}{:>12d}'.format(g.nelem, g.nnode))
    for elem in g.tp:
        f.write('\n')
        for node in elem:
            f.write(' {:>11d}'.format(node + 1))
print('The T file is saved as {}'.format(fileT))

with open(fileB, 'w') as f:
    f.write('{:>12d}'.format(2*(g.yn * g.zn + g.xn * g.zn + g.xn * g.yn)))
    for surf in g.surf:
        for patch in surf:
            f.write('\n')
            for item in patch:
                f.write(' {:>11d}'.format(item + 1))
print('The B file is saved as {}'.format(fileB))

print('                !!!JOB COMPLETE!!!')

