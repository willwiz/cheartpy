#!/usr/bin/env python3

from re import M
import numpy as np
from numpy import sin, cos, array, zeros, sqrt
import argparse
from argparse import RawTextHelpFormatter
from math import pi


parser = argparse.ArgumentParser(description="""
    Makes a cylinder mesh, 2D, using a given mesh with x \in [0,1], y \in [0,1]. Here, x is assumed to be the scaled
    circumferential coordinate and y is the scaled radial coordinate, i.e. \theta = angle * x + \theta_0, r = h*y + h_0.

""", formatter_class=RawTextHelpFormatter)
parser.add_argument('nq', help='how many elements along circumference',
                    type=int)
parser.add_argument('dr', help='the thickness of the hollow circle',
                    type=float)
parser.add_argument('r0', help='the inner radius',
                    type=float)
parser.add_argument('nr', help='how many elements are along the radius',
                    type=int)
parser.add_argument('prefix', help='prefix of the name of the file to export the meshes, _FE.X, _FE.T and _FE.B for be added to outputs',
                    type=str)


# This is gives the order of the nodes in an elent given a starting indice
vorder = [[0, 0], [1, 0], [0, 1], [1, 1]]

# This one has some self wrappiness
class make_grid:
  def __init__(self, arg):
    # extract the arguments first
    self.surf  = []
    self.xsize = float(arg[0])
    self.xn    = int(arg[1])
    self.dx    = float(arg[0])/float(arg[1])
    self.xoff  = float(arg[2])
    self.ysize = float(arg[3])
    self.yn    = int(arg[4])
    self.dy    = float(arg[3])/float(arg[4])
    self.yoff  = float(arg[5])
    # compute the number of nodes and elements
    self.nnode = (self.xn + 1) * (self.yn + 1)
    self.nelem = self.xn * self.yn
    # create empty index array
    self.ind = np.zeros((self.xn + 1, self.yn + 1), dtype=int)
    # fill array
    g = 0
    for j in range(self.yn + 1):
      for i in range(self.xn):
        self.ind[i, j] = g
        g = g + 1
      self.ind[self.xn, j] = self.ind[0, j]
    # create empty space array
    self.space = np.zeros(((self.xn) * (self.yn + 1), 2))
    # fill array
    for j in range(self.yn + 1):
      for i in range(self.xn):
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
    surface[i, 3] = 1
  g.surf.append(surface)
  return

def make_base_grid(nq : int, nr : int) -> make_grid:
  g = make_grid([1.0, nq, 0.0, 1.0, nr, 0.0])
  compute_topology(g)
  compute_surface(g)
  return g

def scaled_cyl_coord(p : np.ndarray, s : np.ndarray) -> np.ndarray:
  q = -2*pi * p[:,0]
  r = s[0] * p[:,1] + s[1]
  return r, q

def cyl_to_cart(r : np.ndarray, q : np.ndarray) -> np.ndarray:
  x = r * cos(q)
  y = r * sin(q)
  return np.stack((x,y)).T

def scaled_cyl_to_cart(p : np.ndarray, s : np.ndarray) -> np.ndarray:
  r, q = scaled_cyl_coord(p, s)
  return cyl_to_cart(r,q)

def create_cylinder(g: make_grid, args) -> None:
  g.space = scaled_cyl_to_cart(g.space, np.array([args.dr, args.r0]))

def main() -> None:
  args=parser.parse_args()
  g = make_base_grid(args.nq, args.nr)
  create_cylinder(g, args)
  fileX = args.prefix + '_FE.X'
  fileT = args.prefix + '_FE.T'
  fileB = args.prefix + '_FE.B'
  with open(fileX, 'w') as f:
    f.write('{:>12d}{:>12d}'.format((g.xn) * (g.yn + 1), 2))
    for [x, y] in g.space:
      # print(x, y)
      f.write('\n{:>26.16E} {:>26.16E}'.format(x, y))
  print('The X file is saved as {}'.format(fileX))

  with open(fileT, 'w') as f:
    f.write('{:>12d}{:>12d}'.format(g.nelem, g.nnode))
    for elem in g.tp:
      f.write('\n')
      for node in elem:
        f.write(' {:>11d}'.format(node + 1))
  print('The T file is saved as {}'.format(fileT))

  with open(fileB, 'w') as f:
    f.write('{:>12d}'.format(2*(g.xn)))
    for surf in g.surf:
      for patch in surf:
        f.write('\n')
        for item in patch:
          f.write(' {:>11d}'.format(item + 1))
  print('The B file is saved as {}'.format(fileB))
  print('                !!!JOB COMPLETE!!!')

if __name__=="__main__":
  main()