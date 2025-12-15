from collections.abc import Sequence
from typing import Literal, Unpack, overload

from ._arg_validation import process_cmdline_args
from ._parser import APIKwargsFind, APIKwargsIndex, CmdLineArgs
from ._parser.main_parser import get_cmd_args, main_parser

__all__ = [
    "cheart2vtu",
    "cheart2vtu_api",
    "cheart2vtu_cli",
    "get_cmd_args",
    "main_parser",
    "process_cmdline_args",
]

def cheart2vtu(cmd_args: CmdLineArgs) -> None: ...
@overload
def cheart2vtu_api(cmd: Literal["find"], **kwargs: Unpack[APIKwargsFind]) -> None: ...
@overload
def cheart2vtu_api(cmd: Literal["index"], **kwargs: Unpack[APIKwargsIndex]) -> None: ...
def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None: ...
