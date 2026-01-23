import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Unpack, overload

from cheartpy.search.trait import AUTO
from pytools.logging import LogLevel

from ._find import find_subparser
from ._index import index_subparser
from ._io import io_parser
from ._settings import setting_parser
from ._topology import find_topology_parser, index_topology_parser
from ._types import (
    SUBPARSER_MODES,
    APIKwargs,
    APIKwargsFind,
    APIKwargsIndex,
    TimeProgArgs,
    VTUProgArgs,
)
from .time_parser import time_parser

if TYPE_CHECKING:
    from collections.abc import Sequence

cheart2vtu_parser = argparse.ArgumentParser()
_subparsers = cheart2vtu_parser.add_subparsers(dest="cmd")
find = _subparsers.add_parser(
    "find",
    help="determine settings automatically",
    parents=[find_subparser, io_parser, find_topology_parser, setting_parser],
)
find.add_argument("var", nargs="*", type=str, help="Optional: variables")
index = _subparsers.add_parser(
    "index",
    help="determine settings automatically",
    parents=[index_subparser, io_parser, index_topology_parser, setting_parser],
)
index.add_argument("var", nargs="*", type=str, help="Optional: variables")


main_parser = argparse.ArgumentParser()
_subparsers = main_parser.add_subparsers(dest="cmd")
find = _subparsers.add_parser(
    "find",
    help="determine settings automatically",
    parents=[find_subparser, io_parser, find_topology_parser, setting_parser],
)
find.add_argument("var", nargs="*", type=str, help="Optional: variables")
index = _subparsers.add_parser(
    "index",
    help="determine settings automatically",
    parents=[index_subparser, io_parser, index_topology_parser, setting_parser],
)
index.add_argument("var", nargs="*", type=str, help="Optional: variables")
time = _subparsers.add_parser(
    "time",
    help="create time series from existing vtu files",
    parents=[time_parser],
)


def get_vtu_cmd_args(args: Sequence[str] | None = None) -> VTUProgArgs:
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
    # If args.cmd is None, print help message and exit
    parsed_args = cheart2vtu_parser.parse_args(args)
    match parsed_args.cmd:
        case "find":
            return VTUProgArgs(**vars(parsed_args))
        case "index":
            return VTUProgArgs(**vars(parsed_args))
        case _:
            cheart2vtu_parser.print_help()
            raise SystemExit(0)


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
    # If args.cmd is None, print help message and exit
    parsed_args = main_parser.parse_args(args)
    match parsed_args.cmd:
        case "find":
            return VTUProgArgs(**vars(parsed_args))
        case "index":
            return VTUProgArgs(**vars(parsed_args))
        case "time":
            return TimeProgArgs(**vars(parsed_args))
        case _:
            main_parser.print_help()
            raise SystemExit(0)


@overload
def get_api_args(cmd: Literal["find"], **kwargs: Unpack[APIKwargsFind]) -> VTUProgArgs: ...
@overload
def get_api_args(cmd: Literal["index"], **kwargs: Unpack[APIKwargsIndex]) -> VTUProgArgs: ...
@overload
def get_api_args(cmd: SUBPARSER_MODES, **kwargs: Unpack[APIKwargs]) -> VTUProgArgs: ...
def get_api_args(cmd: SUBPARSER_MODES, **kwargs: Unpack[APIKwargs]) -> VTUProgArgs:
    match cmd:
        case "find":
            return get_api_args_find(**kwargs)
        case "index":
            if kwargs.get("top") is None:
                msg = "`top` is a required keyword argument for 'index' command"
                raise TypeError(msg)
            return get_api_args_index(**kwargs)  # type: ignore[arg-type]


def get_api_args_find(**kwargs: Unpack[APIKwargsFind]) -> VTUProgArgs:
    mesh_or_top = kwargs.get("mesh", "mesh")
    index = kwargs.get("index", AUTO)
    match kwargs.get("subindex"):
        case "auto":
            subindex = AUTO
        case "none" | None:
            subindex = None
        case (int(i), int(j), int(k)):
            subindex = (i, j, k)
    return VTUProgArgs(
        cmd="find",
        index=index,
        subindex=subindex,
        mesh_or_top=mesh_or_top,
        prefix=kwargs.get("prefix"),
        input_dir=Path(kwargs.get("input_dir", "")),
        output_dir=kwargs.get("output_dir", ""),
        space=kwargs.get("space"),
        boundary=kwargs.get("boundary"),
        prog_bar=kwargs.get("prog_bar", True),
        log=LogLevel[kwargs.get("log", "INFO")],
        binary=kwargs.get("binary", False),
        compress=kwargs.get("compress", True),
        cores=kwargs.get("cores", 1),
        var=kwargs.get("var", []),
    )


def get_api_args_index(**kwargs: Unpack[APIKwargsIndex]) -> VTUProgArgs:
    mesh_or_top = kwargs.get("top")
    index = kwargs.get("index")
    match kwargs.get("subindex"):
        case "auto":
            subindex = AUTO
        case "none" | None:
            subindex = None
        case (int(i), int(j), int(k)):
            subindex = (i, j, k)
    return VTUProgArgs(
        cmd="index",
        index=index,
        subindex=subindex,
        mesh_or_top=mesh_or_top,
        prefix=kwargs.get("prefix"),
        input_dir=Path(kwargs.get("input_dir", "")),
        output_dir=kwargs.get("output_dir", ""),
        space=kwargs.get("space"),
        boundary=kwargs.get("boundary"),
        prog_bar=kwargs.get("prog_bar", True),
        log=LogLevel[kwargs.get("log", "INFO")],
        binary=kwargs.get("binary", False),
        compress=kwargs.get("compress", True),
        cores=kwargs.get("cores", 1),
        var=kwargs.get("var", []),
    )
