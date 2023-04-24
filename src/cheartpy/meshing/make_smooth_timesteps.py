#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Creates a time stepping scheme for a given minimum initial size, number of minimal steps and a maximum size
#  and creates a set of time steps with an exponential and smooth transition
# The inputs of this script are:
#     dt_start n_start n_trans n_total time_end fileout

import os, sys, fileinput

from math import exp
from math import log
from math import floor
from math import ceil
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import Bounds
import time

# These function compares two values and see if they are numerically equal given some numerical error from summing n floats
def same_dbl(A, B, n):
  trial = abs(A - np.full(n, A/n).sum())
  diff = abs(A - B)
  test = max(abs(A), abs(B))
  if (diff <= test * trial):
    return True
  return False

def get_power_size(a, n):
  s, b = 0, a
  for _ in range(0, n):
    s, b = s + b, a*b
  return s

def get_power_max(a, n):
  sum = a
  for _ in range(1, n):
    sum = a*sum
  return sum


def compute_total_time(a, dt0, n0, n1, n2):
  t0 = n0
  t1 = get_power_size(a, n1)
  t2 = get_power_max(a, n1) * n2
  return dt0 * (t0 + t1 + t2)

def find_parameter(dt0, n0, n1, n2, Tf):
  def obf(x):
    a = x[0]
    y = compute_total_time(a, dt0, n0, n1, n2)
    z = y - Tf
    return log(z * z)
  optres = minimize(obf, np.array([1.0]), bounds=Bounds([0.5], [1.001]), method='TNC',tol=1e-14)
  return optres.x

# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
arg = sys.argv

# Check the number of arguments to see if they are correct
#    The number of arguments must be exactly 6
if len(arg) != 7:
  print('{} arguments supplied'.format(len(arg)))
  print('usage: \n\tThere must be exactly 6 arguments:')
  print('\tdt_start n_start n_trans n_total time_end fileout')
  exit()

# Get the filename and open a file
fileout_step = arg[-1] + '.step'
f_step       = open(fileout_step, 'w')
fileout_time = arg[-1] + '.tvi'
f_time       = open(fileout_time, 'w')

# Get the parameters from the inputs
dt0 = float(arg[1])
n0  = int(arg[2])
n1  = int(arg[3])
nt  = int(arg[4])
n2  = nt - n1 - n0
if not (n2 >= 0): raise ValueError('Your total n is less than the number of step in the initial + transition range ... ..., please fix this ... ...')
Tf  = float(arg[5])



# Compute the increase in step size per iteration
par = find_parameter(dt0, n0, n1, n2, Tf)[0]

dt = dt0 * Tf / compute_total_time(par, dt0, n0, n1, n2)
print(dt0, dt)
# Get the time steps in the initial range
list_dt0    = np.full(n0, dt)
# Calculate the remaining time to be accounted for in the later two stages
t_baseline  = nt * dt
# t_remaining = Tf - t_baseline
# if not (t_remaining > 0.0): raise ValueError('Your initial step size are too large, it already exceed the final time in {} steps'.format(nt))

# Compute the remaining list of dts and combine
list_dt1 = dt * np.asarray([par**i for i in range(1, n1+1)])
list_dt2 = dt * np.full(n2, par**n1)
list_dt  = np.concatenate((list_dt0, list_dt1, list_dt2))
Tf_computed = list_dt.sum()

print('A list of dt has been generated with the following:')
print('    {} total time steps and {} time elapse'.format(list_dt.size, list_dt.sum()))
print('    The first  part has {} steps with size {} and {} time elapsed'.format(list_dt0.size, list_dt0[0], list_dt0.sum()))
print('    The second part has {} steps with multiplier of {} and {} time elapsed'.format(list_dt1.size, par, list_dt1.sum()))
print('    The third  part has {} steps with size {} and {} time elapsed'.format(list_dt2.size, list_dt2[0], list_dt2.sum()))
print('Now writing list of dt to file')
if not same_dbl(Tf_computed, Tf, nt): raise ValueError('Total time {} is not the same as inputed {}. I did something wrong in the code!'.format(Tf_computed, Tf))
if not (list_dt.size == nt): raise ValueError('The total number of time steps is not the same as requested, I did something wrong appearantly.')

# Write the total number of time steps
f_step.write('{}'.format(nt))
# Writing the time steps
for i in range(nt):
  f_step.write('\n{} {}'.format(i+1, list_dt[i]))
f_step.close

print('Now writing the list of time elasped to file')
list_t = np.add.accumulate(list_dt)
f_time.write('{}'.format(nt))
# Writing the time steps
for i in range(nt):
  f_time.write('\n{} {}'.format(i+1, list_t[i]))
f_time.close
print('                !!!JOB COMPLETE!!!')







