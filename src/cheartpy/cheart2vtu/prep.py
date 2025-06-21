__all__ = ["init_variable_cache", "parse_cmdline_args"]
import os
from pathlib import Path

import numpy as np
from arraystubs import Arr2
from pytools.logging.trait import ILogger

from cheartpy.cheart_mesh.io import chread_d
from cheartpy.io.indexing.api import get_file_name_indexer
from cheartpy.io.indexing.interfaces import IIndexIterator

from .parser_main import parse_findmode_args, parse_indexmode_args
from .print_headers import print_input_info
from .trait import (
    CheartTopology,
    CmdLineArgs,
    IFormattedName,
    ProgramArgs,
    VariableCache,
)
from .variable_naming import CheartMeshFormat, CheartVarFormat, CheartZipFormat


def parse_cmdline_args(
    args: CmdLineArgs,
    log: ILogger,
) -> tuple[ProgramArgs, IIndexIterator]:
    err: bool = False
    log.info(print_input_info(args))
    # Set the prefix
    if not args.prefix:
        prefix = args.output_dir.replace("_vtu", "") if args.output_dir else "paraview"
    else:
        prefix = args.prefix
    match args.mesh:
        case str():
            x, top, bnd, u = parse_findmode_args(args.mesh)
        case x, t, b:
            x, top, bnd, u = parse_indexmode_args(x, t, b)
    if bnd is not None:
        log.disp(f"Looking for boundary file: {bnd}")
        if os.path.isfile(bnd):
            log.disp(f"<<< Output file name (boundary): {prefix}_boundary.vtu")
        else:
            log.error(f"Boundary file = {bnd} not found.")
            err = True
    else:
        log.disp("<<< No boundary file specified. Skipping boundary export.")
    if args.space is not None:
        name = args.space.split("+")
        if len(name) == 2:
            x, u = name
        else:
            x, u = args.space, None
    # Check if the input and output directory exists
    if not args.input_dir:
        pass
    elif not os.path.isdir(args.input_dir):
        log.error(f"Input folder = {args.input_dir} does not exist")
        err = True
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    # Get the indexer for the variable files
    indexer = get_file_name_indexer(args.index, args.subindex, args.var, args.input_dir)
    if not os.path.isfile(top):
        log.error(f"ERROR: Topology = {top} not found.")
        err = True
    i0 = next(iter(indexer))
    if os.path.isfile(x):
        space = CheartMeshFormat(None, x)
    elif os.path.isfile(f"{x}-{i0}.D"):
        space = CheartVarFormat(args.input_dir, x)
    elif os.path.isfile(f"{x}-{i0}.D.gz"):
        space = CheartZipFormat(args.input_dir, x)
    else:
        space = None
    if u is None:
        disp = None
    elif os.path.isfile(u):
        disp = CheartMeshFormat(None, u)
    elif os.path.isfile(f"{u}-{i0}.D"):
        disp = CheartVarFormat(args.input_dir, u)
    elif os.path.isfile(f"{u}-{i0}.D.gz"):
        disp = CheartZipFormat(args.input_dir, u)
    else:
        log.error(f"Disp = {u} not recognized as mesh, var, or zip")
        raise
    var: dict[str, IFormattedName] = dict()
    for v in args.var:
        if os.path.isfile(os.path.join(args.input_dir, f"{v}-{i0}.D")):
            var[v] = CheartVarFormat(args.input_dir, v)
        elif os.path.isfile(os.path.join(args.input_dir, f"{v}-{i0}.D.gz")):
            var[v] = CheartZipFormat(args.input_dir, v)
        else:
            log.error(f">>>ERROR: Type of {v} cannot be identified.")
            err = True
    if err:
        msg = ">>>ERROR: Some files were not found or could not be identified."
        raise ValueError(msg)
    if space is None:
        log.error(f"Space = {space} not recognized as mesh, var, or zip")
        raise ValueError
    return (
        ProgramArgs(
            prefix,
            args.input_dir,
            args.output_dir,
            args.time_series,
            args.progress_bar,
            args.binary,
            args.compression,
            args.cores,
            top,
            bnd,
            space,
            disp,
            var,
        ),
        indexer,
    )


def init_variable_cache(
    inp: ProgramArgs,
    indexer: IIndexIterator,
) -> VariableCache[np.float64, np.intc]:
    i0 = next(iter(indexer))
    top = CheartTopology(inp.tfile, inp.bfile)
    fx = inp.space[i0]
    space = chread_d(fx)
    if inp.disp is None:
        fd = None
        disp = np.zeros_like(space)
    else:
        fd = inp.disp[i0]
        disp = chread_d(fd)
    x = space + disp
    fv: dict[str, Path] = dict.fromkeys(inp.var.keys(), Path())
    var: dict[str, Arr2[np.float64]] = {}
    for k, fn in inp.var.items():
        name = fn[i0]
        if name.exists():
            fv[k] = name
        else:
            msg = f"initial value for {k} = {name} does not exist"
            raise ValueError(msg)
        var[k] = chread_d(name)
    return VariableCache(top, i0, fx, fd, space, disp, x, fv, var)
