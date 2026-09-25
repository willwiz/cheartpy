from typing import TYPE_CHECKING, NamedTuple, TypedDict, Unpack

from cheartpy.search import (
    DynamicFile,
    FileVariable,
    IIndexIterator,
    create_indexer,
    get_file_type,
)
from pytools.parallel import ThreadMethods
from pytools.result import Err, Ok, Result, all_ok

from ._headers import compose_index_info, format_input_info
from ._struct import ProgramArgs

if TYPE_CHECKING:
    from pathlib import Path

    from pytools.logging import ILogger

    from ._parser import VTUProgArgs


def _get_prefix(args: VTUProgArgs) -> str:
    return args.prefix or args.output_dir.name.replace("_vtu", "") or "paraview"


def _file_check(file: Path | None) -> Result[None]:
    if file is None or file.is_file():
        return Ok(None)
    msg = f"Topology file = {file} does not exist"
    return Err(ValueError(msg))


class _TopFileState(NamedTuple):
    x: FileVariable
    u: FileVariable | None
    t: Path
    b: Path | None


def _check_topology_files(args: VTUProgArgs) -> Result[_TopFileState]:
    match all_ok([_file_check(f) for f in [args.top, args.boundary]]):
        case Ok(): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    if "+" in str(args.space):
        msg = "The space+disp input style is deprecated. Please use -x space -u disp instead."
        return Err(ValueError(msg))
    match get_file_type(args.space, args.input_dir):
        case Ok(space): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    if args.disp:
        match get_file_type(args.disp, args.input_dir):
            case Ok(disp): ...  # fmt: skip
            case Err(e): return Err(e)  # fmt: skip
    else:
        disp = None
    if not (disp is None or isinstance(disp.fname, DynamicFile)):
        msg = "Displacement must be none or changing with time."
        return Err(ValueError(msg))
    return Ok(_TopFileState(space, disp, args.top, args.boundary))


class _MPITypeModeArgs(TypedDict, total=False):
    core: int | None
    thread: int | None


def _parse_mpi_mode(**kwargs: Unpack[_MPITypeModeArgs]) -> ThreadMethods | None:
    if not kwargs:
        return None
    if (n := kwargs.get("core")) is not None:
        return ThreadMethods(core=n)
    if (n := kwargs.get("thread")) is not None:
        return ThreadMethods(thread=n)
    return None


def process_cmdline_args(
    args: VTUProgArgs,
    log: ILogger,
) -> Ok[tuple[ProgramArgs, IIndexIterator]] | Err:
    """Process command line arguments raw into program structs."""
    log.info("Current Run Information:", *format_input_info(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    """x: space, t: topology, b: boundary, u: displacement"""
    match _check_topology_files(args):
        case Ok(mesh): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match all_ok({v: get_file_type(v, args.input_dir) for v in args.point_var}):
        case Ok(point_variables): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match all_ok({v: get_file_type(v, args.input_dir) for v in args.cell_var}):
        case Ok(cell_variables): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match create_indexer(
        {k: v for k, v in {"_X": mesh.x, "_U": mesh.u}.items() if v is not None}
        | dict(point_variables)
        | dict(cell_variables),
        args.index,
        args.subindex,
    ):
        case Ok(indexer):
            ifirst = next(iter(indexer))
        case Err(e):
            return Err(e)
    log.disp(compose_index_info(indexer))
    space = mesh.x.fname if mesh.x.fname.is_dynamic else None
    mpi_mode = _parse_mpi_mode(core=args.core, thread=args.thread)
    options = ProgramArgs(
        prefix=_get_prefix(args),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prog_bar=args.prog_bar,
        binary=args.binary,
        compress=args.compress,
        mpi=mpi_mode,
        tfile=mesh.t,
        bfile=mesh.b,
        xfile=mesh.x.fname[ifirst],
        space=space,
        disp=mesh.u.fname if mesh.u else None,
        cell_var={str(v): v.fname for v in cell_variables.values()},
        point_var={str(v): v.fname for v in point_variables.values()},
    )
    log.debug("Final Arguments:", options=options)
    return Ok((options, indexer))
