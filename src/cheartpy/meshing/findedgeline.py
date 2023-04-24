#!/usr/bin/python3
# -*- coding: utf-8 -*-

# find the segments in a file.B that satisfies the constraints given
#     file.X file.T cons1 cons2 ... consn fileout

import os, sys, fileinput
from math import pi
import numpy as np

origin = np.array([0.0, 8.134, 0.0])

def same_dbl(A, B, n):
  trial = abs(A - np.full(n, A/n).sum())
  diff = abs(A - B)
  test = max(abs(A), abs(B))
  if (diff <= test * trial):
    return True
  return False

def same_dbl_array(A, B, n):
  res = True
  for ai, bi in zip(A, B):
    res = res and same_dbl(ai, bi, n)
  return res


def read_D_array(file):
  with open(file, 'r') as f:
    line     = f.readline().strip().split()
    [n, dim] = [int(item) for item in line[:2]]
    arr      = np.zeros((n, dim))
    try:
      for i in range(n):
        arr[i] = [float(item) for item in f.readline().strip().split()]
    except: raise ValueError('Header information does not match the dimensions of the import')
  return arr, n

def read_T_array(file):
  with open(file, 'r') as f:
    line = f.readline().strip().split()
    n    = int(line[0])
    arr  = []
    try:
      for i in range(n):
        arr.append([int(item) for item in f.readline().strip().split()])
    except: raise ValueError('Header information does not match the dimensions of the import')
  return arr

def filter_points(constraint, space, points):
  func = lambda x, y, z: eval(str(constraint))
  if isinstance(func(0.0, 0.0, 0.0), (bool)):
    points = np.asarray([p for p in points if func(space[p][0], space[p][1], space[p][2])])
  elif isinstance(func(0.0, 0.0, 0.0), (int, float)):
    points = np.asarray([p for p in points if same_dbl(func(space[p][0], space[p][1], space[p][2]), 0.0, 10)])
  else:
    print('The constraint {} given cannot be evaluated!!!'.format(constraint))
    print('Please make sure the inequalities are functions of x, y, z')
    exit()
  return points

def check_edge(arg, space, points):
  for i in range(ncons):
    points = filter_points(arg[i + 3], space, points)
  if (len(points) == 2):
    return True, points
  elif (len(points) > 2):
    return sys.exit('Constraints are not sufficient for narrow down to an edge.')
  return False, points

def find_edge(arg, space, surf):
  collection = []
  for patch in surf:
    err, points = check_edge(arg, space, patch)
    if err: collection.append(points)
  return len(collection), np.asarray(collection)

def theta(point):
  place = (point - origin)[:2]
  if same_dbl_array(point, origin, 10):
    return pi/2
  return np.arctan2(place[0], -place[1])

# def theta_pair(pair):
#   return 0.5*(theta(pair[0]) + theta(pair[1]))
def theta_pair(pair):
  return theta(0.5*(pair[0]+pair[1]))

def clockwise(space, points):
  return False if (space[points[0]][0]*space[points[1]][1] - space[points[1]][0]*space[points[0]][1] >= 0.0) else True

def sort_points(space, points):
  if clockwise(space, points):
    points[:] = points[::-1]
    print('changed to {}'.format(points))
  print(theta(space[points[0]]), theta(space[points[1]]))
  return points

def sort_list(space, points):
  def sorf(x):
    return theta_pair(space[x])
  return sorted(points, key=sorf)


# Get the command line arguments
arg = sys.argv

# Get the number of constraints
ncons = len(arg) - 4
if (ncons < 1): sys.exit('You need at least one constraint!')
fout = open(arg[-1], 'w')

space, npoints = read_D_array(arg[1])
surface = read_T_array(arg[2])
surface = [[i - 1 for i in j] for j in surface]
print('There are ', len(surface),' points.')

n, results = find_edge(arg, space, surface)

results = sort_list(space, results)

print('\n')

for i in range(len(results)):
  results[i] = sort_points(space, results[i])
print(space[results])

results = [[i + 1 for i in j] for j in results]

fout.write('{} {}'.format(len(results), npoints))
for [i1, i2] in results:
  fout.write("\n{} {}".format(i1, i2))
fout.close()