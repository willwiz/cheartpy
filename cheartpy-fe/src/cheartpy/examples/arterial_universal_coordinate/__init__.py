from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from cheartpy.fe.api import create_pfile, create_solver_group, create_time_scheme
from pydantic import BaseModel
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from cheartpy.fe.trait import IPFile


class APIKwargs(TypedDict, total=False):
    output_dir: Path


class TimeOption(BaseModel):
    start: int
    end: int
    dt: float | Path = Path.cwd()


class Options(BaseModel):
    time: TimeOption


def read_options(**kwargs: Unpack[APIKwargs]) -> Options: ...


def uac_pfile(**kwargs: Unpack[APIKwargs]) -> IPFile:
    opts = read_options(**kwargs)
    time = create_time_scheme("time", opts.time.start, opts.time.end, opts.time.dt)
    sg = create_solver_group("sg", time)
    pfile = create_pfile()
    pfile.add_solvergroup(sg)
    pfile.set_outputpath(kwargs.get("output_dir") or Path.cwd())
    return pfile
