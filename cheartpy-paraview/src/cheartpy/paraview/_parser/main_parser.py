import argparse
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Unpack

from cheartpy.io import fix_ch_sfx
from cheartpy.search import SearchMode
from pytools.logging import LogEnum

from ._find import find_subparser
from ._index import index_subparser
from ._io import io_parser
from ._settings import multiprocessing_parser, setting_parser
from ._topology import find_topology_parser, index_topology_parser
from ._types import (
    APIKwargsFind,
    APIKwargsIndex,
    TimeProgArgs,
    VTUProgArgs,
)
from .time_parser import time_parser

if TYPE_CHECKING:
    from collections.abc import Sequence


main_parser = argparse.ArgumentParser()
_subparsers = main_parser.add_subparsers(dest="cmd")
find = _subparsers.add_parser(
    "find",
    help="determine settings automatically",
    parents=[
        find_subparser,
        io_parser,
        find_topology_parser,
        setting_parser,
        multiprocessing_parser,
    ],
)
find.add_argument("--cell-var", nargs="+", type=str, default=[], help="Optional: cell variables")
find.add_argument("point_var", nargs="*", type=str, help="Optional: point variables")
index = _subparsers.add_parser(
    "index",
    help="determine settings automatically",
    parents=[
        index_subparser,
        io_parser,
        index_topology_parser,
        setting_parser,
        multiprocessing_parser,
    ],
)
index.add_argument("--cell-var", nargs="+", type=str, default=[], help="Optional: cell variables")
index.add_argument("point_var", nargs="*", type=str, help="Optional: point variables")
time = _subparsers.add_parser(
    "time",
    help="create time series from existing vtu files",
    parents=[time_parser],
)


def get_cmd_args(args: Sequence[str] | None = None) -> VTUProgArgs | TimeProgArgs:
    """Parse command line arguments.

    Parameters
    ----------
    args : Sequence[str] | None
        List of command line arguments to parse. If None, defaults to sys.argv.

    Returns
    -------
    CmdLineArgs
        Parsed command line arguments as a CmdLineArgs object.

    """
    # Require subparsers to be called, which sets args.cmd
    # If args.cmd is None, display help message and exit
    parsed_args = main_parser.parse_args(args)
    match parsed_args.cmd:
        case "find":
            return get_api_args_find(**vars(parsed_args))
        case "index":
            return get_api_args_index(**vars(parsed_args))
        case "time":
            return TimeProgArgs(**vars(parsed_args))
        case _:
            main_parser.print_help()
            raise SystemExit(0)


class _MeshTopologyFiles(NamedTuple):
    t: Path
    x: Path | str
    b: Path | None
    u: str | None


def _to_path(path: str | Path | None) -> Path | None:
    return Path(path) if path is not None else None


def _parse_findmode_mesh(**kwargs: Unpack[APIKwargsFind]) -> _MeshTopologyFiles:
    mesh = fix_ch_sfx(Path(kwargs.get("mesh", "mesh")))
    space = kwargs.get("space") or mesh.with_suffix(".X")
    boundary = kwargs.get("boundary") or mesh.with_suffix(".B")
    boundary = Path(boundary) if Path(boundary).is_file() else None
    disp = kwargs.get("disp")
    return _MeshTopologyFiles(x=space, u=disp, t=mesh.with_suffix(".T"), b=boundary)


def _parse_indexmode_mesh(**kwargs: Unpack[APIKwargsIndex]) -> _MeshTopologyFiles:
    top = kwargs.get("top")
    space = kwargs.get("space")
    boundary = kwargs.get("boundary")
    disp = kwargs.get("disp")
    return _MeshTopologyFiles(x=space, u=disp, t=Path(top), b=_to_path(boundary))


def get_api_args_find(**kwargs: Unpack[APIKwargsFind]) -> VTUProgArgs:
    mesh = _parse_findmode_mesh(**kwargs)
    index = kwargs.get("index", SearchMode.auto)
    match kwargs.get("subindex"):
        case "auto":
            subindex = SearchMode.auto
        case None:
            subindex = SearchMode.none
        case (int(i), int(j), int(k)):
            subindex = (i, j, k)
    input_dir = Path(kwargs.get("input_dir") or Path.cwd())
    output_dir = _to_path(kwargs.get("output_dir")) or input_dir
    return VTUProgArgs(
        cmd="find",
        index=index,
        subindex=subindex,
        prefix=kwargs.get("prefix"),
        input_dir=input_dir,
        output_dir=output_dir,
        top=mesh.t,
        space=mesh.x,
        disp=mesh.u,
        boundary=mesh.b,
        prog_bar=kwargs.get("prog_bar", True),
        log=LogEnum[kwargs.get("log", "INFO")],
        binary=kwargs.get("binary", False),
        compress=kwargs.get("compress", True),
        core=kwargs.get("core"),
        thread=kwargs.get("thread"),
        interpreter=kwargs.get("interpreter"),
        cell_var=kwargs.get("cell_var", []),
        point_var=kwargs.get("point_var", []),
    )


def get_api_args_index(**kwargs: Unpack[APIKwargsIndex]) -> VTUProgArgs:
    mesh = _parse_indexmode_mesh(**kwargs)
    index = kwargs.get("index", SearchMode.none)
    match kwargs.get("subindex"):
        case "auto":
            subindex = SearchMode.auto
        case None:
            subindex = SearchMode.none
        case (int(i), int(j), int(k)):
            subindex = (i, j, k)
    input_dir = Path(kwargs.get("input_dir") or Path.cwd())
    output_dir = _to_path(kwargs.get("output_dir")) or input_dir
    return VTUProgArgs(
        cmd="index",
        index=index,
        subindex=subindex,
        prefix=kwargs.get("prefix"),
        input_dir=input_dir,
        output_dir=output_dir,
        top=mesh.t,
        space=mesh.x,
        disp=mesh.u,
        boundary=mesh.b,
        prog_bar=kwargs.get("prog_bar", True),
        log=LogEnum[kwargs.get("log", "INFO")],
        binary=kwargs.get("binary", False),
        compress=kwargs.get("compress", True),
        core=kwargs.get("core"),
        thread=kwargs.get("thread"),
        interpreter=kwargs.get("interpreter"),
        cell_var=kwargs.get("cell_var", []),
        point_var=kwargs.get("point_var", []),
    )
