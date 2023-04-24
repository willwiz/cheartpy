#!/usr/bin/env python3

import os
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from numpy import zeros, array, full
from math import floor
from typing import Callable, List
import struct

parser = argparse.ArgumentParser(description="""
      Create scarse array map for mapping variable from linear topologies to quadratic topologies.
      This tool has 2 modes:
        (1) making the map from Lin to Quad topology (--make-map)
        (2) mapping an array from Lin to Quad topology (default)
      --make-map
        Has 2 arguments: linear topology file name, quadratic topology file name
        Also requires 1 cmd arguments: output file name
      default
        Require 2 cmd arguments: file name of map from Lin to Quad, file name of array to be mapped
        Output file name is made by default by appending -quad to the end
        Output file name file name can also be supplied by --prefix
        --index enables batch mode assuming the files have similar names like:
          {name}-#.D
          default output file name is {name}-quad-#.D
      """, formatter_class=RawTextHelpFormatter)
parser.add_argument('--make-map', dest='make_map',   type=str,   nargs=2,  metavar=('lin_map', 'quad_map'),  default=None,
                    help='OPTIONAL: this tool will be set to make the make map mode. Requires the two topology to be supplied')
parser.add_argument('--binary', action='store_true', help='OPTIONAL: assumes that the .D files being imported is binary')
parser.add_argument('--index',      '-i', nargs=3,  dest='index',   type=int,    metavar=('start', 'end', 'step'),
                    help='OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.')
parser.add_argument('--prefix',     '-p', dest='prefix',   type=str,      default=None,
                    help='OPTIONAL: output file will have [tag] appended to the end of name before index numbers and extension')
parser.add_argument('--batch', '-b', dest='batch', action='store_true',
                    help='OPTIONAL: enable looping through multiple variables')
parser.add_argument('name', nargs="+", help='names to files. If --make-map, then last name is the file name, else first name is the map and second name is the input',
                    type=str)

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

def read_arr_int(file):
  arr  = []
  with open(file,'r') as f:
    for line in f:
      arr.append([int(m) for m in line.split()])
  return array(arr)

def read_Tarr(file, n):
  with open(file,'r') as f:
    items = f.readline().strip().split()
    nelem = int(items[0])
    nnode = int(items[1])
    x_arr  = zeros((nelem,n), dtype=int)
    for i in range(nelem):
      items = f.readline().strip().split()
      x_arr[i] = [int(m) for m in items]
  return nnode, array(x_arr)

def read_Darr(file):
  with open(file,'r') as f:
    line  = f.readline().strip()
    items = line.split()
    nnodes = int(items[0])
    dim    = int(items[1])
    x_arr  = zeros((nnodes,dim))
    for i in range(nnodes):
      items = f.readline().strip().split()
      x_arr[i] = [float(m) for m in items]
  return array(x_arr)


def CHRead_d_binary(file : str) -> np.ndarray:
  with open(file, mode='rb') as f:
    nnodes = struct.unpack("i", f.read(4))[0]
    dim    = struct.unpack("i", f.read(4))[0]
    arr    = zeros((nnodes, dim))
    for i in range(nnodes):
      for j in range(dim):
        bite     = f.read(8)
        if (not bite): raise BufferError('Binary buffer being read ran out before indicated range')
        arr[i,j] = struct.unpack("d", bite)[0]
  return arr


def CHWrite_d_binary(file : str, arr : np.ndarray) -> None:
  dim = arr.shape
  with open(file, 'wb') as f:
      f.write(struct.pack('i', dim[0]))
      f.write(struct.pack('i', dim[1]))
      for i in arr:
          for j in i:
              f.write(struct.pack('d', j))
  return

def write_Darr(fout, arr) -> None:
  with open(fout, 'w') as f:
    nnodes, dim = arr.shape
    f.write('{:12d}{:12d}\n'.format(nnodes,dim))
    for vec in arr:
      for x in vec:
        f.write('{:>26.16E}'.format(x))
      f.write('\n')

def write_array_int(file, data):
    with open(file, 'w') as outfile:
        for i in data:
            for j in i:
                outfile.write('{:12d}'.format(j))
            outfile.write('\n')
    return

def edit_val(arr : np.ndarray, ind : int, val : List[int]) -> None:
  if all(a < -1 for a in arr[ind]):
    arr[ind] = val
  elif (set(arr[ind]) == set(val)):
    # print(f"share node index {ind} has value {arr[ind]} which matching {val}")
    pass
  else:
    print(f"index {ind} has value {arr[ind]} which does not match {val}")
    raise LookupError(">>>ERROR: tried to insert index for map which does match input from prior elements")

def gen_map(lin : np.ndarray, quad : np.ndarray, quad_n : int, update : Callable = None) -> np.ndarray:
  rows_lin,  _  = lin.shape
  rows_quad, _ = quad.shape
  try: rows_lin == rows_quad
  except: ValueError('Topologies do not have the same number of elements')
  top_map = full((quad_n, 5), -2, dtype=int)
  try:
    for i in range(rows_quad):
      for j in range(4):
        edit_val(top_map, quad[i,j], [1, lin[i,j], 0, 0, 0])
      edit_val(top_map, quad[i,4], [2, top_map[quad[i,0], 1], top_map[quad[i,1], 1], 0, 0])
      edit_val(top_map, quad[i,5], [2, top_map[quad[i,0], 1], top_map[quad[i,2], 1], 0, 0])
      edit_val(top_map, quad[i,6], [4, top_map[quad[i,0], 1], top_map[quad[i,1], 1], top_map[quad[i,2], 1], top_map[quad[i,3], 1]])
      edit_val(top_map, quad[i,7], [2, top_map[quad[i,1], 1], top_map[quad[i,3], 1], 0, 0])
      edit_val(top_map, quad[i,8], [2, top_map[quad[i,2], 1], top_map[quad[i,3], 1], 0, 0])

      if update is not None:
        update()
  except LookupError as e:
    print(f"fails on element {i + 1} node {j + 1}")
    print(e)

  return top_map



def get_qual_val(map : np.ndarray, arr : np.ndarray) -> np.ndarray:
  if (map[0] < 1 or map[0] > 4): raise AssertionError(f"<<<ERROR: Nodal value must be the average of 1, 2, or 4 other nodes, not {map[0]}")
  res = 0.0
  for i in range(map[0]):
    res = res + arr[map[i + 1]]
  return res/float(map[0])

def lin_to_quad_arr(map : np.ndarray, arr : np.ndarray) -> np.ndarray:
  rows, _ = map.shape
  _, cols = arr.shape
  res = zeros((rows, cols))
  for i, m in enumerate(map):
    res[i] = get_qual_val(m, arr)
  return res

def make_map(args):
  _,  lin_top  = read_Tarr(args.make_map[0], 4)
  nquad, quad_top = read_Tarr(args.make_map[1], 9)
  # Convert to python index
  lin_top  = lin_top - 1
  quad_top = quad_top - 1
  bar = progress_bar(f"Generating Map from {args.make_map[0]} to {args.make_map[1]}", max=len(quad_top))
  top_map = gen_map(lin_top, quad_top, nquad, bar.next)
  write_array_int(args.name[-1], top_map)

def map_vals_i(args, l2q_map, name):
  if (args.prefix is None):
    root, ext = os.path.splitext(name)
    tag = root + "-quad" + ext
  else:
    tag = args.prefix
  # print(tag)
  if args.index==None:
    if args.binary:
      lindata = CHRead_d_binary(name)
    else:
      lindata = read_Darr(name)
    quadata = lin_to_quad_arr(l2q_map, lindata)
    # print(tag)
    if args.binary:
      CHWrite_d_binary(tag, quadata)
    else:
      write_Darr(tag, quadata)
  else:
    bar = progress_bar(f"Interpolating {name} to quad topology", max=floor((args.index[1] - args.index[0])/args.index[2]) + 1)
    for i in range(args.index[0], args.index[1]+args.index[2], args.index[2]):
      fin = name+f'-{i}.D'
      fout = tag+f'-{i}.D'
      if args.binary:
        lindata = CHRead_d_binary(fin)
      else:
        lindata = read_Darr(name)
      quadata = lin_to_quad_arr(l2q_map, lindata)
      if args.binary:
        CHWrite_d_binary(fout, quadata)
      else:
        write_Darr(fout, quadata)
      bar.next()
  print(f"<<<  Job Complete for {name}!")

def map_vals(args):
  if (len(args.name) < 2): raise AssertionError(f"<<<ERROR: normal model requires 2 or 3 arguments: map, file, [filenameout]. {len(args.name)} provided: {args.name}")
  l2q_map = read_arr_int(args.name[0])
  if args.batch:
    for var in args.name[1:]:
      map_vals_i(args, l2q_map, var)
  else:
    map_vals_i(args, l2q_map, args.name[1])

if __name__=="__main__":
  args=parser.parse_args()
  if args.make_map is not None:
    make_map(args)
  else:
    map_vals(args)



