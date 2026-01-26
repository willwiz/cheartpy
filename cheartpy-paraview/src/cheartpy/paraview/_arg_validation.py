from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict, Unpack, overload

from cheartpy.io.api import fix_ch_sfx
from cheartpy.search.api import get_file_name_indexer
from pytools.result import Err, Ok, all_ok

from ._headers import compose_index_info, format_input_info
from ._struct import MPIDef, ProgramArgs
from ._variable_getter import CheartMeshFormat, CheartVarFormat, CheartZipFormat

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cheartpy.search.trait import IIndexIterator
    from pytools.logging import ILogger

    from ._parser.types import SUBPARSER_MODES, VTUProgArgs
    from ._trait import IFormattedName


class _MeshTopologyFiles(NamedTuple):
    x: Path
    t: Path
    b: Path | None
    u: Path | None


def _parse_findmode_args(
    mesh: Path, space: Path | None, bnd: Path | None
) -> Ok[_MeshTopologyFiles] | Err:
    subs = fix_ch_sfx(mesh)
    space = space or (subs.with_suffix(".X"))
    match space.name.split("+"):
        case (str(),):
            x, u = space, None
        case str(_space), str(_disp):
            x, u = space.parent / _space, space.parent / _disp
        case _:
            msg = "Invalid space name format, expected 'name' or 'name+disp'."
            return Err(ValueError(msg))
    t = subs.with_suffix(".T")
    if not t.is_file():
        msg = f"Mesh topology file = {t} not found."
        return Err(ValueError(msg))
    if isinstance(bnd, Path) and not bnd.is_file():
        msg = f"Boundary file = {bnd} not found."
        return Err(ValueError(msg))
    b = bnd or (subs.with_suffix(".B"))
    b = b if b.is_file() else None
    return Ok(_MeshTopologyFiles(space or x, t, b, u))


def _parse_indexmode_args(
    top: Path, space: Path | None, bnd: Path | None
) -> Ok[_MeshTopologyFiles] | Err:
    if space is None:
        msg = "In index mode, space name must be provided."
        return Err(ValueError(msg))
    match space.name.split("+"):
        case (str(),):
            x, u = space, None
        case str(_space), str(_disp):
            x = space.parent / _space
            u = space.parent / _disp
        case _:
            msg = "Invalid space name format, expected 'name' or 'name+disp'."
            return Err(ValueError(msg))
    if not top.is_file():
        msg = f"Topology file = {top} not found."
        return Err(ValueError(msg))
    if bnd and not bnd.is_file():
        msg = f"Boundary file = {bnd} not found."
        return Err(ValueError(msg))
    return Ok(_MeshTopologyFiles(x, top, bnd, u))


_MESH_FILE_PARSER: Mapping[
    SUBPARSER_MODES, Callable[[Path, Path | None, Path | None], Ok[_MeshTopologyFiles] | Err]
] = {
    "find": _parse_findmode_args,
    "index": _parse_indexmode_args,
}


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


def _get_mesh_names(
    args: VTUProgArgs,
) -> Ok[_MeshTopologyFiles] | Err:
    match _MESH_FILE_PARSER[args.cmd](args.mesh_or_top, args.space, args.boundary):
        case Ok(mesh):
            return Ok(mesh)
        case Err(e):
            return Err(e)


@overload
def _check_variable_format(u: None, first: str | int, root: Path | None = None) -> Ok[None]: ...
@overload
def _check_variable_format(
    u: Path, first: str | int, root: Path | None = None
) -> Ok[IFormattedName] | Err: ...
@overload
def _check_variable_format(
    u: str, first: str | int, root: Path | None = None
) -> Ok[IFormattedName] | Err: ...
def _check_variable_format(
    u: Path | str | None,
    first: str | int,
    root: Path | None = None,
) -> Ok[IFormattedName] | Ok[None] | Err:
    match u:
        case None:
            return Ok(None)
        case Path():
            u = (root / u) if root else u
        case str():
            u = (root / u) if root else Path(u)
    if u.is_file():
        return Ok(CheartMeshFormat(u.parent, u.name))
    if (u.parent / f"{u.name}-{first}.D").is_file():
        return Ok(CheartVarFormat(u.parent, u.name))
    if (u.parent / f"{u.name}-{first}.D.gz").is_file():
        return Ok(CheartZipFormat(u.parent, u.name))
    msg = f"Variable {u} not recognized as one of:"
    msg += f" Mesh = {u}"
    msg += f" Var  = {u.parent / f'{u.name}-{first}.D'}"
    msg += f" Zip  = {u.parent / f'{u.name}-{first}.D.gz'}"
    return Err(ValueError(msg))


def find_variable_formats(
    x: Path,
    u: Path | None,
    variables: Sequence[str],
    ifirst: str | int,
    input_dir: Path,
) -> Ok[tuple[IFormattedName, IFormattedName | None, Sequence[IFormattedName]]] | Err:
    match _check_variable_format(x, ifirst):
        case Ok(space):
            ...
        case Err(e):
            return Err(e)
    match _check_variable_format(u, ifirst):
        case Ok(disp):
            ...
        case Err(e):
            return Err(e)
    match all_ok([_check_variable_format(v, ifirst, input_dir) for v in variables]):
        case Ok(var):
            ...
        case Err(e):
            return Err(e)
    return Ok((space, disp, var))


class _MPITypeModeArgs(TypedDict, total=False):
    core: int | None
    thread: int | None


def _parse_mpi_mode(**kwargs: Unpack[_MPITypeModeArgs]) -> MPIDef | None:
    if not kwargs:
        return None
    if (n := kwargs.get("core")) is not None:
        return MPIDef("core", n)
    if (n := kwargs.get("thread")) is not None:
        return MPIDef("thread", n)
    return None


def process_cmdline_args(
    args: VTUProgArgs,
    log: ILogger,
) -> Ok[tuple[ProgramArgs, IIndexIterator]] | Err:
    """Process command line arguments raw into program structs."""
    log.info(*format_input_info(args))
    prefix = _get_prefix(args)
    match _check_dirs_inputs(args):
        case Ok((input_dir, output_dir)):
            pass
        case Err(e):
            return Err(e)
    match get_file_name_indexer(args.index, args.subindex, args.var, root=input_dir, log=log):
        case Ok(indexer):
            ifirst = next(iter(indexer))
        case Err(e):
            return Err(e)
    log.disp(compose_index_info(indexer))
    """x: space, t: topology, b: boundary, u: displacement"""
    match _get_mesh_names(args):
        case Ok((x, top, bnd, u)):
            if bnd is None:
                log.disp("<<< No boundary file specified/found.")
        case Err(e):
            return Err(e)
    match find_variable_formats(x, u, args.var, ifirst, input_dir):
        case Ok((xfile, disp, var)):
            pass
        case Err(e):
            return Err(e)
    space = None if isinstance(xfile, CheartMeshFormat) else xfile
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
                tfile=top,
                bfile=bnd,
                xfile=xfile[ifirst],
                space=space,
                disp=disp,
                var={v.name: v for v in var},
            ),
            indexer,
        )
    )
