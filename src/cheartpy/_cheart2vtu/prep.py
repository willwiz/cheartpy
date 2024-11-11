__all__ = ["parse_cmdline_args", "init_variable_cache"]
import os

# import numpy as np
from typing import Sequence
from ..io.indexing import IIndexIterator, get_file_name_indexer
from ..tools.basiclogging import BLogger, ILogger
from .interfaces import *
from .variable_naming import *
from .fio import *

# from .third_party import compress_vtu
# from concurrent import futures
# from ..var_types import i32, f64, Arr
# from ..cheart_mesh.io import *
# from ..xmlwriter.xmlclasses import XMLElement, XMLWriters

# # from ..cheart2vtu_core.print_headers import print_input_info
# # from ..cheart2vtu_core.main_parser import get_cmdline_args
# # from ..cheart2vtu_core.file_indexing import (
# #     IndexerList,
# #     get_file_name_indexer,
# # )
# # from ..cheart2vtu_core.data_types import (
# #     CheartMeshFormat,
# #     CheartVarFormat,
# #     CheartZipFormat,
# #     InputArguments,
# #     ProgramArgs,
# #     VariableCache,
# #     CheartTopology,
# # )
# from ..tools.progress_bar import ProgressBar
# from ..tools.parallel_exec import *
from .parser_main import get_cmdline_args
from .print_headers import *


def parse_cmdline_args(
    cmd_args: Sequence[str | float | int] | None = None, LOG: ILogger = BLogger("INFO")
) -> tuple[ProgramArgs, IIndexIterator]:
    err: bool = False
    args = get_cmdline_args([str(v) for v in cmd_args] if cmd_args else None)
    print_input_info(args)
    if not args.input_folder:
        pass
    elif not os.path.isdir(args.input_folder):
        LOG.error(f"Input folder = {args.input_folder} does not exist")
        err = True
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
    indexer = get_file_name_indexer(
        args.index, args.subindex, args.var, args.input_folder
    )
    if not os.path.isfile(args.tfile):
        LOG.error(f"ERROR: Topology = {args.tfile} not found.")
        err = True
    i0 = next(iter(indexer))
    space = None
    if os.path.isfile(args.xfile):
        space = CheartMeshFormat(args.xfile)
    elif os.path.isfile(f"{args.xfile}-{i0}.D"):
        space = CheartVarFormat(args.input_folder, args.xfile)
    elif os.path.isfile(f"{args.xfile}-{i0}.D.gz"):
        space = CheartZipFormat(args.input_folder, args.xfile)
    disp = None
    if args.disp is None:
        pass
    elif os.path.isfile(args.disp):
        disp = CheartMeshFormat(args.disp)
    elif os.path.isfile(f"{args.disp}-{i0}.D"):
        disp = CheartVarFormat(args.input_folder, args.disp)
    elif os.path.isfile(f"{args.disp}-{i0}.D.gz"):
        disp = CheartZipFormat(args.input_folder, args.disp)
    else:
        LOG.error(f"Disp = {args.disp} not recognized as mesh, var, or zip")
        err = True
    var: dict[str, IFormattedName] = dict()
    for v in args.var:
        if os.path.isfile(os.path.join(args.input_folder, f"{v}-{i0}.D")):
            var[v] = CheartVarFormat(args.input_folder, v)
        elif os.path.isfile(os.path.join(args.input_folder, f"{v}-{i0}.D.gz")):
            var[v] = CheartZipFormat(args.input_folder, v)
        else:
            print(f">>>ERROR: Type of {v} cannot be identified.")
            err = True
    if err:
        raise ValueError("At least one error was triggered.")
    if space is None:
        LOG.error(f"Space = {args.xfile} not recognized as mesh, var, or zip")
        raise
    return (
        ProgramArgs(
            args.prefix,
            args.input_folder,
            args.output_folder,
            args.time_series,
            args.progress_bar,
            args.binary,
            args.compression,
            args.cores,
            args.tfile,
            args.bfile,
            space,
            disp,
            var,
        ),
        indexer,
    )


def init_variable_cache(inp: ProgramArgs, indexer: IIndexIterator) -> VariableCache:
    i0 = next(iter(indexer))
    top = CheartTopology(inp.tfile, inp.bfile)
    fx = inp.space[i0]
    fd = None if inp.disp is None else inp.disp[i0]
    nodes = get_space_data(fx, fd)
    var: dict[str, str] = dict.fromkeys(inp.var.keys(), "")
    for v, fn in inp.var.items():
        name = fn[i0]
        if os.path.isfile(name):
            var[v] = name
        else:
            raise ValueError(f"The initial value for {v} cannot be found")
    return VariableCache(i0, top, nodes, fx, fd, var)
