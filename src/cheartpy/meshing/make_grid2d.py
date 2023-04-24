#!/usr/bin/python3
# -*- coding: utf-8 -*-

# This creates an hexahedral mesh of uniform grids
# The inputs of this script are:
#     x_length x_n x_offset y_length y_n y_offset fileout

import os, sys, fileinput
import numpy as np
import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description="""
      To be made
      """, formatter_class=RawTextHelpFormatter)
parser.add_argument('xlen',  metavar='X Length', type=float,
                    help='Length in the X direction')
parser.add_argument('xn',    metavar='X Elem',   type=int,
                    help='Number of elements in the X direction')
parser.add_argument('xoff',  metavar='X Offset', type=float,
                    help='How much to shift the mesh in the X direction')
parser.add_argument('ylen',  metavar='Y Length', type=float,
                    help='Length in the Y direction')
parser.add_argument('yn',    metavar='Y Elem',   type=int,
                    help='Number of elements in the Y direction')
parser.add_argument('yoff',  metavar='Y Offset', type=float,
                    help='How much to shift the mesh in the Y direction')
parser.add_argument('prefix', type=str,
                    help='Prefix of the files to save the mesh to')

# This is gives the order of the nodes in an elent given a starting indice
vorder = [[0, 0], [1, 0], [0, 1], [1, 1]]


class make_grid:
    def __init__(self, arg):
        self.tp   = []
        self.tpi  = []
        self.surf = []
        self.edge = []
        self.ns   = 0
        # extract the arguments first
        self.xsize = arg.xlen
        self.xn    = arg.xn
        self.dx    = self.xsize/float(self.xn)
        self.xoff  = arg.xoff
        self.ysize = arg.ylen
        self.yn    = arg.yn
        self.dy    = self.ysize/float(self.yn)
        self.yoff  = arg.yoff
        # compute the number of nodes and elements
        self.nnode = (self.xn + 1) * (self.yn + 1)
        self.nelem = self.xn * self.yn
        # create empty index array
        self.ind = np.zeros((self.xn + 1, self.yn + 1), dtype=int)
        # fill array
        g = 0
        for j in range(self.yn + 1):
            for i in range(self.xn + 1):
                self.ind[i, j] = g
                g = g + 1
        # create empty space array
        self.space = np.zeros((self.nnode, 2))
        # fill array
        for j in range(self.yn + 1):
            for i in range(self.xn + 1):
                self.space[self.ind[i,j]] = [i*self.dx + self.xoff, j*self.dy + self.yoff]
    def __getitem__(self, key):
        # if key is of invalid type or value, the list values will raise the error
        return self.space[self.ind[key]]

def compute_topology(g):
    g.tp = np.zeros((g.nelem, 4), dtype=int)
    n    = 0
    for j in range(g.yn):
        for i in range(g.xn):
            for m in range(4):
                g.tp[n, m] = g.ind[i + vorder[m][0], j + vorder[m][1]]
            n = n + 1
    g.tpi = np.zeros((g.xn, g.yn), dtype=int)
    l = 0
    for j in range(g.yn):
        for i in range(g.xn):
            g.tpi[i, j] = l
            l = l + 1
    return

def compute_surface(g):
    # for the -y surface
    surface = np.zeros((g.xn, 4), dtype=int)
    for i in range(g.xn):
        # Get the element #
        surface[i, 0] = g.tpi[i, 0]
        # Get the two node #
        surface[i, 1] = g.ind[i,     0]
        surface[i, 2] = g.ind[i + 1, 0]
        # label the patch, note the indexing is 1 less than in cheart
        surface[i, 3] = 0
        g.ns = g.ns + 1
    g.surf.append(surface)
    # for the +x surface
    surface = np.zeros((g.yn, 4), dtype=int)
    for j in range(g.yn):
        # Get the element #
        surface[j, 0] = g.tpi[g.xn - 1,  j]
        # Get the two node #
        surface[j, 1] = g.ind[g.xn,  j]
        surface[j, 2] = g.ind[g.xn,  j + 1]
        # label the patch, note the indexing is 1 less than in cheart
        surface[j, 3] = 1
        g.ns = g.ns + 1
    g.surf.append(surface)
    # for the +y surface
    surface = np.zeros((g.xn, 4), dtype=int)
    for i in range(g.xn):
        # Get the element #
        surface[i, 0] = g.tpi[g.xn - 1 - i, g.yn - 1]
        # Get the two node #
        surface[i, 1] = g.ind[g.xn - i, g.yn]
        surface[i, 2] = g.ind[g.xn - i - 1,     g.yn]
        # label the patch, note the indexing is 1 less than in cheart
        surface[i, 3] = 2
        g.ns = g.ns + 1
    g.surf.append(surface)
    # for the -x surface
    surface = np.zeros((g.yn, 4), dtype=int)
    for j in range(g.yn):
        # Get the element #
        surface[j, 0] = g.tpi[0,  g.yn - 1 - j]
        # Get the two node #
        surface[j, 1] = g.ind[0,  g.yn - j    ]
        surface[j, 2] = g.ind[0,  g.yn - j - 1]
        # label the patch, note the indexing is 1 less than in cheart
        surface[j, 3] = 3
        g.ns = g.ns + 1
    g.surf.append(surface)
    return



def compute_pads(g):
    if (g.xn % 10) == 0:
        xl = g.xn//10
    else:
        raise ValueError('xn not divisible by 10')
    if (g.yn % 10) == 0:
        yl = g.yn//10
    else:
        raise ValueError('yn not divisible by 10')
    # for the -y surface
    surface = np.zeros((6*xl, 4), dtype=int)
    m = 0
    for i in [1, 2, 4, 5, 7, 8]:
        for k in range(xl):
            # Get the element #
            surface[m, 0] = g.tpi[xl*i + k, 0]
            # Get the two node #
            surface[m, 1] = g.ind[xl*i + k,     0]
            surface[m, 2] = g.ind[xl*i + k + 1, 0]
            # label the patch, note the indexing is 1 less than in cheart
            surface[m, 3] = 4
            m = m + 1
    g.surf.append(surface)
    # for the +x surface
    surface = np.zeros((6*yl, 4), dtype=int)
    m = 0
    for j in [1, 2, 4, 5, 7, 8]:
        for k in range(yl):
            # Get the element #
            surface[m, 0] = g.tpi[g.xn - 1,  yl*j + k]
            # Get the two node #
            surface[m, 1] = g.ind[g.xn,  yl*j + k]
            surface[m, 2] = g.ind[g.xn,  yl*j + k + 1]
            # label the patch, note the indexing is 1 less than in cheart
            surface[m, 3] = 5
            m = m + 1
    g.surf.append(surface)
    # for the +y surface
    surface = np.zeros((6*xl, 4), dtype=int)
    m = 0
    for i in [1, 2, 4, 5, 7, 8]:
        for k in range(xl):
            # Get the element #
            surface[m, 0] = g.tpi[g.xn - 1 - (xl*i + k), g.yn - 1]
            # Get the two node #
            surface[m, 1] = g.ind[g.xn - (xl*i + k), g.yn]
            surface[m, 2] = g.ind[g.xn - (xl*i + k) - 1,     g.yn]
            # label the patch, note the indexing is 1 less than in cheart
            surface[m, 3] = 6
            m = m + 1
    g.surf.append(surface)
    # for the -x surface
    surface = np.zeros((6*yl, 4), dtype=int)
    m = 0
    for j in [1, 2, 4, 5, 7, 8]:
        for k in range(yl):
            # Get the element #
            surface[m, 0] = g.tpi[0,  g.yn - 1 - (yl*j + k)]
            # Get the two node #
            surface[m, 1] = g.ind[0,  g.yn - (yl*j + k)]
            surface[m, 2] = g.ind[0,  g.yn - (yl*j + k) - 1]
            # label the patch, note the indexing is 1 less than in cheart
            surface[m, 3] = 7
            m = m + 1
    g.surf.append(surface)
    return



def main(args=None):
    args=parser.parse_args(args=args)
    fileX = args.prefix + '_FE.X'
    fileT = args.prefix + '_FE.T'
    fileB = args.prefix + '_FE.B'
    g = make_grid(args)
    print('Made a grid with {} nodes'.format(g.nnode))
    compute_topology(g)
    print('Created a topology for the grid with {} elements'.format(g.nelem))
    compute_surface(g)
    print('Generated a the 6 surfaces facing each side')
    # compute_pads(g)
    with open(fileX, 'w') as f:
        f.write('{:>12d}{:>12d}'.format(g.nnode, 2))
        for [x, y] in g.space:
            f.write('\n{:18} {:18}'.format(x, y))
    print('The X file is saved as {}'.format(fileX))
    with open(fileT, 'w') as f:
        f.write('{:>12d}{:>12d}'.format(g.nelem, g.nnode))
        for elem in g.tp:
            f.write('\n')
            for node in elem:
                f.write(' {:>11d}'.format(node + 1))
    print('The T file is saved as {}'.format(fileT))
    with open(fileB, 'w') as f:
        # f.write('{:>12d}'.format(2*(g.yn + g.xn + g.xn//10 * 6 + g.yn//10 * 6)))
        f.write('{:>12d}'.format(g.ns))
        for surf in g.surf:
            for patch in surf:
                f.write('\n')
                for item in patch:
                    f.write(' {:>11d}'.format(item + 1))
    print('The B file is saved as {}'.format(fileB))
    print('                !!!JOB COMPLETE!!!')

if __name__=="__main__":
    main()