#!/usr/bin/env python3

import os, sys
import argparse
from argparse import RawTextHelpFormatter

def split(f:str):
  name, _ = os.path.splitext(f)
  path, name = os.path.split(name)
  return path, name

def gen_script(args, rest):
  import importlib
  path, name = split(args.pfile)
  sys.path.append(path)
  pkg = importlib.import_module(name)
  p = pkg.get_PFile()
  fout = name + '.P'
  with open(fout,'w') as f:
    p.write(f)

def run_script(args, rest):
  import importlib
  import subprocess as sp
  path, name = split(args.pfile)
  sys.path.append(path)
  pkg = importlib.import_module(name)
  p = pkg.get_PFile()
  fout = name + '.P'
  with open(fout,'w') as f:
    p.write(f)
  if args.cores > 1:
    cmd = ['mpiexec', '-n', f'{args.cores!s}', 'cheartsolver.out', fout, *rest]
  else:
    cmd = ['cheartsolver.out', fout, *rest]
  print(" ".join(cmd))
  if args.log == 'Null':
    sp.run(cmd, stdout=sp.DEVNULL)
  elif args.log is not None:
    with open(args.log, 'w') as f:
      sp.run(cmd, stdout=f, stderr=sp.STDOUT)
  else:
    sp.run(cmd)
  if not(args.keep):
    os.remove(fout)


def run_pfile(args, rest):
  import subprocess as sp
  fout = args.pfile
  if args.cores > 1:
    cmd = ['mpiexec', '-n', f'{args.cores!s}', 'cheartsolver.out', fout, *rest]
  else:
    cmd = ['cheartsolver.out', fout, *rest]
  print(" ".join(cmd))
  if args.log == 'Null':
    sp.run(cmd, stdout=sp.DEVNULL)
  elif args.log is not None:
    with open(args.log, 'w') as f:
      sp.run(cmd, stdout=f, stderr=sp.STDOUT)
  else:
    sp.run(cmd)


def is_exe(fpath):
  return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
  fpath, fname = os.path.split(program)
  if fpath:
    if is_exe(program):
        return program
  else:
    for path in os.environ["PATH"].split(os.pathsep):
      exe_file = os.path.join(path, program)
      if is_exe(exe_file):
        return exe_file
  return None





def cheart_usage():
  print("""\033[4mHelp Options\033[0m:
    ./cheartsolver.out --version  --revision
    ./cheartsolver.out --help
          : provides a list of all flags and their details
    ./cheartsolver.out --help=list_flags
          : provides a concise list of all flags (alternative is --help=nano )
    ./cheartsolver.out --help=list_scopes
          : provides a concise list of all scopes which can be accessed
    ./cheartsolver.out --help=list_problem_types
          : provides a concise list of all problem classes which can be accessed
    ./cheartsolver.out --help=flag
          : provides details about a specific flag (excluding the !), e.g. --help=UseBasis
    ./cheartsolver.out --help=scope
          : provides details about flags that fall within a specific scope, e.g. --help=basis
    ./cheartsolver.out --parse-macros
  """)
  print("""\033[4mExecuting Cheart\033[0m:
    ./cheartsolver.out PROBLEM_FILE.P

    ./cheartsolver.out PROBLEM_FILE.P --no-output  -n
          : restricts output to only force output commands
    ./cheartsolver.out PROBLEM_FILE.P --output-level [=min,std,verbose,nag]
          : controls the level of information exported during solve, e.g. --output-level=verbose
    ./cheartsolver.out PROBLEM_FILE.P --log
          : exports output to a log file
    ./cheartsolver.out PROBLEM_FILE.P --log-distributed
          : exports output to a log file (by rank)
    ./cheartsolver.out PROBLEM_FILE.P --dump-residual
          : exports residual values (by variable in *.D files)
    ./cheartsolver.out PROBLEM_FILE.P --dump-matrix [=matlab,cheart]
          : exports matrix and residual data (default cheart format)
    ./cheartsolver.out PROBLEM_FILE.P --dump-intermediate
          : exports intermediate solutions computed during nonlinear solve
    ./cheartsolver.out PROBLEM_FILE.P --dump-memory
          : prints current memory usage per process used to the terminal at different stages in the program
    ./cheartsolver.out PROBLEM_FILE.P --pedantic-printing
          : prints additional information for each solver iteration, including more detailed residual and change in variable values
    ./cheartsolver.out PROBLEM_FILE.P --data-assimilation

    ./cheartsolver.out PROBLEM_FILE.P -#MACRO1=Value1 -#MACRO2=Value2
          : run the problem with additional #MACROs, if the same #MACRO exists in the P-file, the one defined in the command line takes precedence
  """)

parser = argparse.ArgumentParser(
  prog='Python Cheart Pfile Interface',
  description=
"""
To be made
""", formatter_class=RawTextHelpFormatter)

def self_help(args=None, rest=None):
  parser.print_help()

parser.set_defaults(main=self_help)
parser.add_argument('--cheart-usage', action='store_true', help='print the usage information for cheartsolver.out')
subparsers = parser.add_subparsers(help='To be made')
parser_gen = subparsers.add_parser('gen', help='generate a Pfile only')
parser_gen.add_argument('pfile', type=str)
parser_run = subparsers.add_parser('run', help='run cheart with a py script instead of a Pfile')
parser_run.add_argument('pfile', type=str)
parser_run.add_argument('--keep', action='store_true',
                    help='OPTIONAL: keep pfile after')
parser_run.add_argument('--log', type=str,
                    help='OPTIONAL: store output in a log file with given name')
parser_run.add_argument('--cores', '-n',  type=int, default=1,
                    help='OPTIONAL: number of cores to compute with')
parser_pf = subparsers.add_parser('pfile', help='run cheart with a Pfile')
parser_pf.add_argument('pfile', type=str)
parser_pf.add_argument('--log', type=str,
                    help='OPTIONAL: store output in a log file with given name')
parser_pf.add_argument('--cores', '-n',  type=int, default=1,
                    help='OPTIONAL: number of cores to compute with')
parser_gen.set_defaults(main=gen_script)
parser_run.set_defaults(main=run_script)
parser_pf.set_defaults(main=run_pfile)

def main_cli(args=None):
  args, rest = parser.parse_known_args(args=None)
  args.main(args, rest)

if __name__=="__main__":
  main_cli()
