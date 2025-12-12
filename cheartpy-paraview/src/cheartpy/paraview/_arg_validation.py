from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypeIs, overload

from cheartpy.io.api import fix_ch_sfx
from cheartpy.search.api import get_file_name_indexer
from pytools.result import Err, Ok

from ._headers import print_input_info
from ._variable_getter import CheartMeshFormat, CheartVarFormat, CheartZipFormat
from .struct import ProgramArgs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cheartpy.search.trait import IIndexIterator
    from pytools.logging.trait import ILogger

    from ._parser import SUBPARSER_MODES, CmdLineArgs
    from ._trait import IFormattedName


class _MeshTopologyFiles(NamedTuple):
    x: str
    t: Path
    b: Path | None
    u: str | None


def _parse_findmode_args(
    mesh: str, space: str | None, bnd: Path | None
) -> Ok[_MeshTopologyFiles] | Err:
    subs: str = fix_ch_sfx(mesh)
    space = space or (subs + "X")
    match space.split("+"):
        case (str(_space),):
            x, u = _space, None
        case str(_space), str(_disp):
            x, u = _space, _disp
        case _:
            msg = "Invalid space name format, expected 'name' or 'name+disp'."
            return Err(ValueError(msg))
    t = Path(subs + "T")
    if not t.is_file():
        msg = f"Mesh topology file = {t} not found."
        return Err(ValueError(msg))
    b = Path(bnd or (subs + "B"))
    b = b if b.is_file() else None
    return Ok(_MeshTopologyFiles(space or x, t, b, u))


def _parse_indexmode_args(
    top: str, space: str | None, bnd: Path | None
) -> Ok[_MeshTopologyFiles] | Err:
    if not space:
        msg = "In index mode, space name must be provided."
        return Err(ValueError(msg))
    spacename: list[str] = space.split("+")
    match spacename:
        case (str(_space),):
            x, u = _space, None
        case str(_space), str(u):
            x = _space
        case _:
            msg = "Invalid space name format, expected 'name' or 'name+disp'."
            return Err(ValueError(msg))
    if not (t := Path(top)).is_file():
        msg = f"Topology file = {t} not found."
        return Err(ValueError(msg))
    if bnd and not bnd.is_file():
        msg = f"Boundary file = {bnd} not found."
        return Err(ValueError(msg))
    return Ok(_MeshTopologyFiles(x, t, bnd, u))


_MESH_FILE_PARSER: Mapping[
    SUBPARSER_MODES, Callable[[str, str | None, Path | None], Ok[_MeshTopologyFiles] | Err]
] = {
    "find": _parse_findmode_args,
    "index": _parse_indexmode_args,
}


def _get_prefix(args: CmdLineArgs) -> str:
    if args.prefix:
        return args.prefix
    return Path(args.output_dir).name.replace("_vtu", "") if args.output_dir else "paraview"


def _check_dirs_inputs(args: CmdLineArgs) -> Ok[tuple[Path, Path]] | Err:
    if not args.input_dir.is_dir():
        msg = f"Input folder = {args.input_dir} does not exist"
        return Err(ValueError(msg))
    output_dir = Path(args.output_dir) if args.output_dir else Path()
    output_dir.mkdir(exist_ok=True)
    return Ok((args.input_dir, output_dir))


def _get_mesh_names(
    args: CmdLineArgs,
) -> Ok[_MeshTopologyFiles] | Err:
    match _MESH_FILE_PARSER[args.cmd](args.mesh_or_top, args.space, args.boundary):
        case Ok(mesh):
            return Ok(mesh)
        case Err(e):
            return Err(e)


@overload
def _check_variable_format(u: None, first: str | int, root: Path) -> None: ...
@overload
def _check_variable_format(u: str, first: str | int, root: Path) -> IFormattedName | ValueError: ...
def _check_variable_format(
    u: str | None,
    first: str | int,
    root: Path,
) -> IFormattedName | ValueError | None:
    if u is None:
        return u
    if (root / u).is_file():
        return CheartMeshFormat(root, u)
    if (root / f"{u}-{first}.D").is_file():
        return CheartVarFormat(root, u)
    if (root / f"{u}-{first}.D.gz").is_file():
        return CheartZipFormat(root, u)
    msg = f"Variable {u} not recognized as mesh, var, or zip"
    return ValueError(msg)


# def _check_boundary_file(
#     bnd: Path | str | None,
#     prefix: str,
#     log: ILogger,
# ) -> Path | None | ValueError:
#     if bnd is None:
#         log.disp("<<< No boundary file specified. Skipping boundary export.")
#         return None
#     bnd = Path(bnd)
#     log.info(f"Looking for boundary file: {bnd}")
#     if bnd.is_file():
#         log.disp(f"<<< Output file name (boundary): {prefix}_boundary.vtu")
#         return bnd

#     log.info(f"Boundary file = {bnd} not found.")
#     return ValueError()


# def _check_for_file(file: Path | str, msg: str) -> Path | ValueError:
#     file = Path(file)
#     if not file.is_file():
#         msg = msg.format(file=file)
#         return ValueError(msg)
#     return file


def _capture_err[T](var: T | ValueError, log: ILogger) -> TypeIs[T]:
    """Capture errors in variable formats."""
    if isinstance(var, ValueError):
        log.error(var)
        return False
    return True


def _capture_err_sequence[T](
    var: tuple[T | ValueError, ...],
    log: ILogger,
) -> TypeIs[tuple[T, ...]]:
    """Capture errors in variable formats."""
    return all(_capture_err(v, log) for v in var)


def process_cmdline_args(
    args: CmdLineArgs,
    log: ILogger,
) -> Ok[tuple[ProgramArgs, IIndexIterator]] | Err:
    """Process command line arguments raw into program structs."""
    log.info(*print_input_info(args))
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
    """x: space, t: topology, b: boundary, u: displacement"""
    match _get_mesh_names(args):
        case Ok((x, top, bnd, u)):
            log.disp("<<< No boundary file specified. Skipping boundary export.")
        case Err(e):
            return Err(e)
    space = _check_variable_format(x, ifirst, Path())
    disp = _check_variable_format(u, ifirst, Path())
    var = tuple([_check_variable_format(v, ifirst, input_dir) for v in args.var])
    if not (
        _capture_err(space, log)
        and _capture_err(disp, log)
        and _capture_err(top, log)
        and _capture_err(bnd, log)
        and _capture_err_sequence(var, log)
    ):
        return Err(ValueError("Invalid command line arguments"))
    return Ok(
        (
            ProgramArgs(
                prefix=prefix,
                input_dir=input_dir,
                output_dir=output_dir,
                prog_bar=args.prog_bar,
                binary=args.binary,
                compress=args.compress,
                cores=args.cores,
                tfile=top,
                bfile=bnd,
                xfile=space,
                disp=disp,
                var={v.name: v for v in var},
            ),
            indexer,
        )
    )
