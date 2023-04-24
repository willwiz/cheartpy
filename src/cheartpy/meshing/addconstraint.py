#!/usr/bin/python
# -*- coding: utf-8 -*-

# Takes a .T file for a line and produce a list of nodes on the line so that boundary conditions can be applied.


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

arg = sys.argv

if len(arg) < 4:
  print('usage: \n\t./addconstraint.py Linefile.Pt val1 val2 ... valn fileout')
  exit()

filelin = arg[1]
fileout = '{}'.format(arg[-1])
fout = open(fileout, 'w')

# Get the number of constraints
ncons = len(arg) - 3

# First import the mesh data for the line
print('Processing the Ptfile data.')


with open(filelin, 'r') as f:
  line = f.readline()
  fout.write(line)
  line = line.strip()
  items = line.split()
  b_elements = int(items[0])
  print('There are {} elements:'.format(b_elements))
  bar = progress_bar('Processing', max=b_elements)
  for line in f:
    fout.write('{} '.format(line.strip()))
    for i in range(ncons):
      fout.write('{} '.format(arg[i + 2]))
    fout.write('\n')
    bar.next()
  bar.finish()

fout.close()
print('Operation complete!\n')