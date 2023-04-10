#!/usr/bin/env python3

import os, sys
import importlib
import subprocess as sp
import argparse
from argparse import RawTextHelpFormatter

def split(f:str):
  name, _ = os.path.splitext(f)
  path, name = os.path.split(name)
  return path, name

def gen_script(args, rest):
  path, name = split(args.pfile)
  sys.path.append(path)
  pkg = importlib.import_module(name)
  p = pkg.get_PFile()
  fout = name + '.P'
  with open(fout,'w') as f:
    p.write(f)

def run_script(args, rest):
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

parser = argparse.ArgumentParser(
  prog='Python Cheart Pfile Interface',
  description=
"""
To be made
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('--prefix', type=str, default=None)
subparsers = parser.add_subparsers(help='To be made')
parser_gen = subparsers.add_parser('gen', help='generate a Pfile only')
parser_gen.add_argument('pfile', type=str)
parser_run = subparsers.add_parser('run', help='run cheart with a py script in place of a Pfile')
parser_run.add_argument('pfile', type=str)
parser_run.add_argument('--keep', action='store_true',
                    help='OPTIONAL: keep pfile after')
parser_run.add_argument('--log', type=str,
                    help='OPTIONAL: store output in a log file with given name')
parser_run.add_argument('--cores', '-n',  type=int, default=1,
                    help='OPTIONAL: number of cores to compute with')
parser_pf = subparsers.add_parser('pfile', help='run cheart with a py script in place of a Pfile')
parser_pf.add_argument('pfile', type=str)
parser_pf.add_argument('--log', type=str,
                    help='OPTIONAL: store output in a log file with given name')
parser_pf.add_argument('--cores', '-n',  type=int, default=1,
                    help='OPTIONAL: number of cores to compute with')
parser_gen.set_defaults(main=gen_script)
parser_run.set_defaults(main=run_script)
parser_pf.set_defaults(main=run_pfile)

if __name__=="__main__":
  args, rest = parser.parse_known_args(args=None)
  args.main(args, rest)
