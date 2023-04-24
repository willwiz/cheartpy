#!/usr/bin/python
# -*- coding: utf-8 -*-

# Creates a time stepping scheme for the given input
# The input of the script are:
#     n1 size1 n2 size2 ... nn sizen fileout

import os, sys, fileinput
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

# Check for the number of arguments
#   The number of arguments must by at least 3, n, size, filename
if len(arg) < 4:
  print('usage: \n\t The must have at least 3 arguments n, step size, filename \n\t./create_timestep.py n_1 size_1 ... n_n size_n filename_out')
  exit()

if len(arg) % 2 != 0:
  print('The number of arguments is incorrect. \n\t usage: \n\t./create_timestep.py n_1 size_1 ... n_n size_n filename_out')
  exit()

# Get the filename to be saved to
fileout = arg[-1]
f = open(fileout, 'w')

# Count the number of segments
n_seg = len(arg)/2 - 1

# Convert the inputs to usable format
stepn = [int(i) for i in arg[1:-2:2]]
stepsize = [float(i) for i in arg[2::2]]

# Count the total number of time steps
n_total = sum(stepn)
# Write the total
f.write('{}'.format(n_total))

# Print information on the inputs
print('There are {:d} segments:'.format(n_seg))
for seg in range(n_seg):
  print('\tsegment {}:\tn={}\tsize={}'.format(seg+1, arg[2 * seg + 1], arg[2 * seg + 2]))

# Start the progress bar
bar = progress_bar('Processing', max=n_total)

# Writing the time steps
i = 0 # This is the time step number
for n in range(n_seg):
  for j in range(stepn[n]):
    i = i + 1   # update the step number
    f.write('\n{} {}'.format(i, stepsize[n]))
    bar.next()  # advance the progress bar
f.close()

print('\nJOB COMPLETE!!!')
