import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack, overload

from cheartpy.search.trait import AUTO
from pytools.logging import LogLevel

from . import SUBPARSER_MODES, APIKwargs, APIKwargsFind, APIKwargsIndex, CmdLineArgs
from ._find import find_subparser
from ._index import index_subparser
from ._io import io_parser
from ._settings import setting_parser
from ._topology import find_topology_parser, index_topology_parser

if TYPE_CHECKING:
    from collections.abc import Sequence

main_parser = argparse.ArgumentParser()
subparsers = main_parser.add_subparsers(dest="cmd")
find = subparsers.add_parser(
    "find",
    help="determine settings automatically",
    parents=[find_subparser, io_parser, find_topology_parser, setting_parser],
)
find.add_argument("var", nargs="*", type=str, help="Optional: variables")
index = subparsers.add_parser(
    "index",
    help="determine settings automatically",
    parents=[index_subparser, io_parser, index_topology_parser, setting_parser],
)
index.add_argument("var", nargs="*", type=str, help="Optional: variables")


class _DefaultArgs(TypedDict):
    prefix: str | None
    input_dir: Path
    output_dir: str
    space: str | None
    boundary: Path | None
    prog_bar: bool
    log: LogLevel
    binary: bool
    compress: bool
    cores: int
    var: Sequence[str]


_DEFAULT_ARGS: _DefaultArgs = {
    "prefix": None,
    "input_dir": Path(),
    "output_dir": "",
    "space": None,
    "boundary": None,
    "prog_bar": True,
    "log": LogLevel.INFO,
    "binary": False,
    "compress": True,
    "cores": 1,
    "var": [],
}


def get_cmd_args(args: Sequence[str] | None = None) -> CmdLineArgs:
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
    parsed_args = main_parser.parse_args(
        args,
        namespace=CmdLineArgs(
            None,  # pyright: ignore[reportArgumentType]
            None,  # NOTE: default from parser
            None,
            mesh_or_top=None,  # pyright: ignore[reportArgumentType], NOTE: default from parser
            **_DEFAULT_ARGS,
        ),
    )
    if parsed_args.cmd is None:  # pyright: ignore[reportUnnecessaryComparison]
        main_parser.print_help()
        raise SystemExit(0)
    if parsed_args.mesh_or_top is None:  # pyright: ignore[reportUnnecessaryComparison]
        print(f"Argparse failed, got mesh_or_top: {parsed_args.mesh_or_top}")
        raise SystemExit(1)
    if not parsed_args.mesh_or_top:
        msg: str = "Never: Mesh or topology file is a required arg. [Unreachable]"
        raise AssertionError(msg)
    return parsed_args


@overload
def get_api_args(cmd: Literal["find"], **kwargs: Unpack[APIKwargsFind]) -> CmdLineArgs: ...
@overload
def get_api_args(cmd: Literal["index"], **kwargs: Unpack[APIKwargsIndex]) -> CmdLineArgs: ...
@overload
def get_api_args(cmd: SUBPARSER_MODES, **kwargs: Unpack[APIKwargs]) -> CmdLineArgs: ...
def get_api_args(cmd: SUBPARSER_MODES, **kwargs: Unpack[APIKwargs]) -> CmdLineArgs:
    match kwargs.get("subindex"):
        case "auto":
            subindex = AUTO
        case "none" | None:
            subindex = None
        case (int(i), int(j), int(k)):
            subindex = (i, j, k)
    match cmd:
        case "find":
            mesh_or_top = kwargs.get("mesh", "mesh")
            index = kwargs.get("index", AUTO)
        case "index":
            mesh_or_top = kwargs.get("top")
            if mesh_or_top is None:
                msg = "`top` is a required keyword argument."
                raise TypeError(msg)
            index = kwargs.get("index")
    return CmdLineArgs(
        cmd=cmd,
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
