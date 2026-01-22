from collections.abc import Sequence
from typing import Literal, Unpack, overload

from ._parser import APIKwargsFind, APIKwargsIndex, CmdLineArgs
from ._time_series import (
    create_time_series_api,
    create_time_series_cli,
    create_time_series_core,
    create_time_series_json,
)

__all__ = [
    "cheart2vtu",
    "cheart2vtu_api",
    "cheart2vtu_cli",
    "create_time_series_api",
    "create_time_series_cli",
    "create_time_series_core",
    "create_time_series_json",
]

def cheart2vtu(cmd_args: CmdLineArgs) -> None: ...
@overload
def cheart2vtu_api(cmd: Literal["find"], **kwargs: Unpack[APIKwargsFind]) -> None: ...
@overload
def cheart2vtu_api(cmd: Literal["index"], **kwargs: Unpack[APIKwargsIndex]) -> None: ...
def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None: ...
