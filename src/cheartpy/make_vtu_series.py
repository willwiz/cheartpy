#/bin/python
#
# pvpython script to convert CHeart data to vtk unstructured grid format (vtu)
#
# author: Will Zhang (willwz@gmail.com)
#
# Usage: prefix  start  end  step
# Options:
#     --time   / -t : indicate a time file to add
#     --folder / -f : indicate a folder to work from
#     --name   / -n : indicate a output filename
import os.path
import numpy as np
import numpy.typing as npt
import sys
import typing as tp
import argparse
import dataclasses
import re
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

################################################################################################
# The argument parse
parser = argparse.ArgumentParser(description='converts cheart output Dfiles into vtu files with time steps for paraview')
parser.add_argument('--make-time-series', dest='time_series', default=None,
                         type=str, help='OPTIONAL: incorporate time data, supply a file for the time step.')
parser.add_argument('--index',         '-i', nargs=3, dest='irange',    action='store', default=[0,0,1],
                    type=int, metavar=('start', 'end', 'step'), help='MANDATORY: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.')

parser.add_argument('--folder',     '-f', dest='outfolder',   action='store', default=None,
                    type=str, help='supply a name for the folder to store the vtu outputs')
parser.add_argument('prefix', action='store', type=str, metavar=('prefix'),
                    help='supply a name for the folder to store the vtu outputs')


@dataclasses.dataclass
class InputArgs:
  prefix : str
  i0 : int
  it : int
  di : int
  folder : tp.Optional[str] = None
  time : tp.Dict[int, float] = dataclasses.field(default_factory=dict)
  nt : int = 0


def xml_write_header(f):
  f.write('<?xml version="1.0"?>\n')
  f.write('<VTKFile type="Collection" version="0.1"\n')
  f.write('         byte_order="LittleEndian"\n')
  f.write('         compressor="vtkZLibDataCompressor">\n')
  f.write('  <Collection>\n')

def xml_write_footer(f):
  f.write('  </Collection>\n')
  f.write('</VTKFile>\n')

def xml_write_content(f, item, time):
  f.write('    <DataSet timestep="{}" group="" part="0"\n'.format(time))
  f.write('             file="{}"/>\n'.format(item))


def import_time_data(file:str) -> tp.Tuple[int , tp.Dict[int, float]]:
  arr  = dict()
  with open(file, 'r') as f:
    try:
      n = int(f.readline.strip())
    except ValueError as e:
      print(">>>ERROR: check file format. Time series data has 1 int for header.")
      raise e
    except Exception as e:
      raise e
    arr[0] = 0.0
    try:
      for i in range(n):
        s, v = f.readline().strip().split()
        arr[int(s)] = float(v) + arr[i]
    except Exception as e:
      raise e
  return len(arr), arr


def check_args(args) -> InputArgs:
  inp = InputArgs(args.prefix, args.irange[0], args.irange[1], args.irange[2], args.folder)
  if args.time_series is None:
    for i in range(inp.i0, inp.it, inp.di):
      inp.time[i] = float(i)
    inp.nt = len(inp.time)
  else:
    inp.nt, inp.time = import_time_data(args.time_series)
  file_check=False
  for i in range(inp.i0, inp.it, inp.di):
    if not os.path.isfile(os.path.join(args.folder, f"{args.prefix}-{i}.vtu")):
      print(f'>>>ERROR: {args.prefix}-{i}.vtu cannot be found')
      file_check = True
    if not i in inp.time:
      print(f'>>>ERROR: step {i} is not in the time step file')
      file_check = True
  if file_check:
    raise FileNotFoundError()
  print("All files are found.")




def print_cmd_header(inp:InputArgs):
  print("################################################################################################")
  print("    script for putting together a collection with the time serie added")
  print("################################################################################################")
  print("")
  print("<<< Output folder:                                   {}".format(inp.folder))
  print("<<< Input file name prefix:                          {}".format(inp.prefix))
  print("<<< Data series:                                     From {} to {} with increment of {}".format(inp.i0, inp.it, inp.di))
  print("<<< Output file name:                                {}".format(inp.prefix+".pvd"))
  print("")




def main():
  args = parser.parse_args()
  inp = check_args(args)
  print_cmd_header(inp)
  bar = progress_bar('Processing', max=inp.nt)
  with open(os.path.join(inp.folder, inp.prefix+".pvd"), "w") as f:
    xml_write_header(f)
    for i in range(inp.i0, inp.it, inp.di):
      xml_write_content(f, os.path.join(inp.folder, f"{inp.prefix}-{i}.vtu"), inp.time[i])
      bar.next()
    xml_write_footer(f)
  bar.finish()
  print("    The process is complete!")
  print("################################################################################################")


if __name__ == "__main__":
    main()

