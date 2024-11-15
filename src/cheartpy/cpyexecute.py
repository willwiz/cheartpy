#!/usr/bin/env python3

import os
import sys
import subprocess as sp
import argparse
from .cheart.p_file import PFile


def split_filename(f: str):
    name, _ = os.path.splitext(f)
    pack = ".".join(name.split(os.sep))
    path, name = os.path.split(name)
    return path, name, pack


def gen_script(args, rest):
    import importlib

    path, name, pack = split_filename(args.pfile)
    sys.path.append(os.getcwd())
    pkg = importlib.import_module(pack)
    p = pkg.get_PFile()
    fout = name + ".P"
    with open(fout, "w") as f:
        p.write(f)


def run_cheartpy(
    p: PFile, fout: str, *rest, cores: int = 1, log=None, keep=False
) -> None:
    """Must give a unique strain token via the `fout` parameter"""
    with open(fout, "w") as f:
        p.write(f)
    if cores > 1:
        cmd = ["mpiexec", "-n", f"{cores!s}", "cheartsolver.out", fout, *rest]
    else:
        cmd = ["cheartsolver.out", fout, *rest]
    print(" ".join(cmd))
    if log == "Null":
        sp.run(cmd, stdout=sp.DEVNULL)
    elif log is not None:
        with open(log, "w") as f:
            sp.run(cmd, stdout=f, stderr=sp.STDOUT)
    else:
        sp.run(cmd)
    if not (keep):
        os.remove(fout)


def run_script(args, rest):
    import importlib
    import subprocess as sp

    path, name, pack = split_filename(args.pfile)
    sys.path.append(os.getcwd())
    pkg = importlib.import_module(pack)
    p = pkg.get_PFile()
    fout = name + ".P"
    run_cheartpy(p, fout, *rest, cores=args.cores, log=args.log, keep=args.keep)


def run_pfile(args, rest):
    import subprocess as sp

    fout = args.pfile
    if args.cores > 1:
        cmd = ["mpiexec", "-n", f"{args.cores!s}", "cheartsolver.out", fout, *rest]
    else:
        cmd = ["cheartsolver.out", fout, *rest]
    print(" ".join(cmd))
    if args.log == "Null":
        sp.run(cmd, stdout=sp.DEVNULL)
    elif args.log is not None:
        with open(args.log, "w") as f:
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


def cheart_help(args: argparse.Namespace, rest=None):
    sp.run(["cheartsolver.out", f"--help={args.string}"])


def run_prep(args: argparse.Namespace, rest: list[str]):
    import importlib
    import subprocess as sp

    path, name, pack = split_filename(args.pyfile)
    sys.path.append(os.getcwd())
    pkg = importlib.import_module(pack)
    p = pkg.get_PFile()
    fout = name + ".P"
    with open(fout, "w") as f:
        p.write(f)
    cmd = ["cheartsolver.out", fout, "--prep", *rest]
    print(" ".join(cmd))
    if args.log == "Null":
        sp.run(cmd, stdout=sp.DEVNULL)
    elif args.log is not None:
        with open(args.log, "w") as f:
            sp.run(cmd, stdout=f, stderr=sp.STDOUT)
    else:
        sp.run(cmd)
    if not (args.keep):
        os.remove(fout)


def self_default(args, rest: list[str]):
    sp.run(["cheartsolver.out", *rest])


parser = argparse.ArgumentParser(
    prog="cpyexec", description="""Python Cheart Pfile Interface"""
)

parser.set_defaults(main=self_default)

subparsers = parser.add_subparsers()
parser_help = subparsers.add_parser(
    "help", help="call cheart --help=var", description="call cheart --help=var"
)
parser_help.add_argument("string", type=str)
parser_help.set_defaults(main=cheart_help)
parser_gen = subparsers.add_parser(
    "gen",
    help="generate a Pfile from py script",
    description="generate a Pfile from py script",
)
parser_gen.add_argument("pfile", type=str)
parser_gen.set_defaults(main=gen_script)
parser_run = subparsers.add_parser(
    "py",
    help="run cheart with a py script instead of a Pfile",
    description="run cheart from py script",
)
parser_run.add_argument("pfile", type=str)
parser_run.add_argument(
    "--keep", action="store_true", help="OPTIONAL: keep pfile after"
)
parser_run.add_argument(
    "--log", type=str, help="OPTIONAL: store output in a log file with given name"
)
parser_run.add_argument(
    "--cores",
    "-n",
    type=int,
    default=1,
    help="OPTIONAL: number of cores to compute with",
)
parser_run.set_defaults(main=run_script)
parser_prep = subparsers.add_parser(
    "prep",
    help="run cheart prep but with a py script",
    description="run cheart prep but with a py script",
)
parser_prep.add_argument("pyfile", type=str)
parser_prep.add_argument(
    "--keep", action="store_true", help="OPTIONAL: keep pfile after"
)
parser_prep.add_argument(
    "--log", type=str, help="OPTIONAL: store output in a log file with given name"
)
parser_prep.set_defaults(main=run_prep)
parser_pf = subparsers.add_parser(
    "run", help="run cheart with a Pfile", description="run cheart with a Pfile"
)
parser_pf.add_argument("pfile", type=str)
parser_pf.add_argument(
    "--log", type=str, help="OPTIONAL: store output in a log file with given name"
)
parser_pf.add_argument(
    "--cores",
    "-n",
    type=int,
    default=1,
    help="OPTIONAL: number of cores to compute with",
)
parser_pf.set_defaults(main=run_pfile)


def main_cli(args=None):
    args, rest = parser.parse_known_args(args=None)
    args.main(args, rest)


if __name__ == "__main__":
    main_cli()
