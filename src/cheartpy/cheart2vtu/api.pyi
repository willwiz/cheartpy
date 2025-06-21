__all__ = ["cheart2vtu", "cheart2vtu_api", "cheart2vtu_cli"]
from collections.abc import Sequence
from typing import Literal

from pytools.logging.trait import LogLevel

from .trait import CmdLineArgs

def cheart2vtu(cmd_args: CmdLineArgs) -> None: ...
def cheart2vtu_api(
    prefix: str | None = None,
    index: tuple[int, int, int] | None = None,
    subindex: tuple[int, int, int] | Literal["auto", "none"] | None = "none",
    variables: Sequence[str] = ...,
    input_dir: str = "",
    output_dir: str = "",
    mesh: str | tuple[str, str, str] = "mesh",
    space: str | None = None,
    time_series: str | None = None,
    binary: bool = False,
    compression: bool = True,
    progress_bar: bool = True,
    cores: int = 1,
    log: LogLevel = ...,
) -> None: ...
def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None: ...
