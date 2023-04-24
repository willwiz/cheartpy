#!/usr/bin/python
# -*- coding: utf-8 -*-

# find the nodes in a file.X that satisfies the constraints given
# The inputs of this script are:
#     filein cons1 cons2 ... consn fileout

import os, sys, fileinput
import numpy as np


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*', printEnd = "\r"):
  """
  Call in a loop to create terminal progress bar
  @params:
      iteration   - Required  : current iteration (Int)
      total       - Required  : total iterations (Int)
      prefix      - Optional  : prefix string (Str)
      suffix      - Optional  : suffix string (Str)
      decimals    - Optional  : positive number of decimals in percent complete (Int)
      length      - Optional  : character length of bar (Int)
      fill        - Optional  : bar fill character (Str)
      printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
  """
  percent = ("{0:." + str(decimals) + "f}").format(100 if (total == 0) else 100 * (iteration / float(total)))
  filledLength = int(length if (total == 0) else length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
  # Print New Line on Complete
  if iteration == total:
      print()

class progress_bar:
  def __init__(self, message, max=100):
    self.n = max
    self.i = 0
    self.message = message
    printProgressBar(self.i, self.n, prefix = self.message, suffix = 'Complete', length = 50)
  def next(self):
    self.i = self.i + 1
    printProgressBar(self.i, self.n, prefix = self.message, suffix = 'Complete', length = 50)
  def finish(self):
    printProgressBar(self.n, self.n, prefix = self.message, suffix = 'Complete', length = 50)
# Get the precision for double below
dbl_eps = 4.4408920985e-15



# These function compares two values and see if they are numerically equal
def same_dbl(A):
  if (abs(A) < dbl_eps):
    return True
  return False



# Get the command line arguments
arg = sys.argv

# Get the number of constraints
ncons = len(arg) - 3

# Check the number of arguments
# The number of arguments must be 3 or more
if len(arg) < 1:
  print('{} arguments supplied.'.format(len(arg) - 1))
  print('There must be at least 3 arguments')
  print('usage:')
  print('\tfilein cons1 cons2 ... consn fileout')
  exit()

# Get the filename and open a file
filein = arg[1]
fin = open(filein, 'r')

# Get the filename and open a file
fileout = arg[-1]
fout = open(fileout, 'w')

# Read in the constraints
with open(filein, 'r') as f:
    line = f.readline()
    line = line.strip()
    items = line.split()
    nodes_total = int(items[0])
    print('There are {} points in the X file'.format(nodes_total))
    collect = []
    bar = progress_bar('Importing data', max=nodes_total)
    try:
        for line in f:
            line = line.strip()
            items = [float(i) for i in line.split()]
            collect.append(items)
            bar.next()
    except:
        print
    bar.finish()
n_points = len(collect)

# Check to see if the X file has errors
if (nodes_total != n_points):
    print('The number of nodes indicated in the header of the X file {} \
        is not equal to the number of points following the header {}'.format(nodes_total, n_points))
    exit()

# Generate a list of points cooresponding to the nodes
points = range(n_points)

# Start eliminating the points based on the constraints in order
for i in range(ncons):
    # Set up a work array
    func = lambda x, y, z: eval(str(arg[i + 2]))
    if isinstance(func(0.0, 0.0, 0.0), (bool)):
        points = [p for p in points if func(collect[p][0], collect[p][1], collect[p][2])]
    elif isinstance(func(0.0, 0.0, 0.0), (int, float)):
        points = [p for p in points if same_dbl(func(collect[p][0], collect[p][1], collect[p][2]))]
    else:
        print('The {}th constraint given cannot be evaluated!!!'.format(i))
        print('Please make sure the inequalities are functions of x, y, z')
        exit()
    n_points = len(points)
    print('Array has been filtered by the {}th constraint'.format(i))

print('\nNow writing to file.')
fout.write('{}'.format(n_points))
for i in range(n_points):
    fout.write('\n{}'.format(points[i] + 1))
print('The filter array has been written to {}'.format(fileout))


