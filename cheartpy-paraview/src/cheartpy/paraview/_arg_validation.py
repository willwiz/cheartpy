from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict, Unpack

from cheartpy.search._varible_index import (
    SingleFile,
    TimeSeriesFile,
    VarType,
    create_indexer,
    get_file_type,
)
from pytools.parallel import ThreadMethods
from pytools.result import Err, Ok, Result, all_ok

from ._headers import compose_index_info, format_input_info
from ._struct import ProgramArgs

if TYPE_CHECKING:
    from cheartpy.search import IIndexIterator
    from pytools.logging import ILogger

    from ._parser.types import VTUProgArgs


def _get_prefix(args: VTUProgArgs) -> str:
    if args.prefix:
        return args.prefix
    return args.output_dir.name.replace("_vtu", "") if args.output_dir else "paraview"


def _check_dirs_inputs(args: VTUProgArgs) -> Ok[tuple[Path, Path]] | Err:
    if not args.input_dir.is_dir():
        msg = f"Input folder = {args.input_dir} does not exist"
        return Err(ValueError(msg))
    output_dir = Path(args.output_dir) if args.output_dir else Path()
    output_dir.mkdir(exist_ok=True)
    return Ok((args.input_dir, output_dir))


class _TopFileState(NamedTuple):
    x: VarType
    u: VarType | None
    t: Path
    b: Path | None


def _check_topology_files(args: VTUProgArgs) -> Result[_TopFileState]:
    if not args.top.is_file():
        msg = f"Topology file = {args.top} does not exist"
        return Err(ValueError(msg))
    if args.boundary and not args.boundary.is_file():
        msg = f"Boundary file = {args.boundary} does not exist"
        return Err(ValueError(msg))
    if "+" in str(args.space):
        msg = "The space+disp input style is deprecated. Please use -x space -u disp instead."
        return Err(ValueError(msg))
    match get_file_type(args.space, args.input_dir):
        case Ok(space):
            ...
        case Err(e): return Err(e)  # fmt: skip
    if args.disp:
        match get_file_type(args.disp, args.input_dir):
            case Ok(disp):
                ...
            case Err(e): return Err(e)  # fmt: skip
    else:
        disp = None
    if not (disp is None or isinstance(disp.fname, TimeSeriesFile)):
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
    prefix = _get_prefix(args)
    match _check_dirs_inputs(args):
        case Ok((input_dir, output_dir)): ...  # fmt: skip
        case Err(e):
            return Err(e)
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
    match create_indexer({**point_variables, **cell_variables}, args.index, args.subindex):
        case Ok(indexer):
            ifirst = next(iter(indexer))
        case Err(e):
            return Err(e)
    log.disp(compose_index_info(indexer))
    match mesh.x.fname:
        case SingleFile():
            xfile = mesh.x.fname[ifirst]
            space = None
        case TimeSeriesFile():
            xfile = mesh.x.fname[ifirst]
            space = mesh.x.fname
    mpi_mode = _parse_mpi_mode(core=args.core, thread=args.thread)
    return Ok(
        (
            ProgramArgs(
                prefix=prefix,
                input_dir=input_dir,
                output_dir=output_dir,
                prog_bar=args.prog_bar,
                binary=args.binary,
                compress=args.compress,
                mpi=mpi_mode,
                tfile=mesh.t,
                bfile=mesh.b,
                xfile=xfile,
                space=space,
                disp=mesh.u.fname if mesh.u else None,
                cell_var={str(v): v.fname for v in cell_variables.values()},
                point_var={str(v): v.fname for v in point_variables.values()},
            ),
            indexer,
        )
    )
