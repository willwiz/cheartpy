from __future__ import annotations

__all__ = ["process_cmdline_args"]
from pathlib import Path
from typing import TYPE_CHECKING, TypeIs, overload

from cheartpy.cheart_mesh.io import fix_suffix
from cheartpy.io.indexing.api import get_file_name_indexer

from ._headers import print_input_info
from ._variable_getter import CheartMeshFormat, CheartVarFormat, CheartZipFormat
from .struct import CmdLineArgs, IFormattedName, ProgramArgs

if TYPE_CHECKING:
    from pytools.logging.trait import ILogger

    from cheartpy.io.indexing.interfaces import IIndexIterator


def _parse_findmode_args(mesh: str) -> tuple[str, str, str | None, None]:
    subs: str = fix_suffix(mesh)
    space = subs + "X"
    topology = subs + "T"
    boundary = subs + "B"
    boundary = boundary if Path(boundary).exists() else None
    return space, topology, boundary, None


def _parse_indexmode_args(x: str, t: str, b: str) -> tuple[str, str, str | None, str | None]:
    spacename: list[str] = x.split("+")
    match spacename:
        case (str(s),):
            space, disp = s, None
        case str(s), str(u):
            space, disp = s, u
        case _:
            msg = "Invalid space name format, expected 'name' or 'name+disp'."
            raise ValueError(msg)
    return space, t, b, disp


def _get_prefix(args: CmdLineArgs) -> str:
    if args.prefix:
        return args.prefix
    return Path(args.output_dir).name.replace("_vtu", "") if args.output_dir else "paraview"


def _check_dirs_inputs(args: CmdLineArgs) -> tuple[Path, Path] | tuple[ValueError, Path]:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        msg = f"Input folder = {args.input_dir} does not exist"
        return ValueError(msg), Path()
    output_dir = Path(args.output_dir) if args.output_dir else Path()
    output_dir.mkdir(exist_ok=True)
    return input_dir, output_dir


def _get_mesh_names(args: CmdLineArgs) -> tuple[str | ValueError, str, str | None, str | None]:
    match args.mesh:
        case str():
            x, top, bnd, u = _parse_findmode_args(args.mesh)
        case x, t, b:
            x, top, bnd, u = _parse_indexmode_args(x, t, b)
    if args.space is not None:
        match args.space.split("+"):
            case str(x), str(u):
                space, disp = x, u
            case (str(x),):
                space, disp = x, None
            case _:
                msg = "Invalid space name format, expected 'name' or 'name+disp'."
                return ValueError(msg), top, bnd, None
    else:
        space, disp = x, u
    return space, top, bnd, disp


@overload
def _check_variable_format(u: None, first: str | int, root: Path) -> None: ...
@overload
def _check_variable_format(u: ValueError, first: str | int, root: Path) -> ValueError: ...
@overload
def _check_variable_format(u: str, first: str | int, root: Path) -> IFormattedName | ValueError: ...
def _check_variable_format(
    u: ValueError | str | None,
    first: str | int,
    root: Path,
) -> IFormattedName | ValueError | None:
    if u is None:
        return u
    if isinstance(u, ValueError):
        return u
    if (root / u).is_file():
        return CheartMeshFormat(root, u)
    if (root / f"{u}-{first}.D").is_file():
        return CheartVarFormat(root, u)
    if (root / f"{u}-{first}.D.gz").is_file():
        return CheartZipFormat(root, u)
    msg = f"Variable {u} not recognized as mesh, var, or zip"
    return ValueError(msg)


def _check_boundary_file(
    bnd: Path | str | None,
    prefix: str,
    log: ILogger,
) -> Path | None | ValueError:
    if bnd is None:
        log.disp("<<< No boundary file specified. Skipping boundary export.")
        return None
    bnd = Path(bnd)
    log.info(f"Looking for boundary file: {bnd}")
    if bnd.is_file():
        log.disp(f"<<< Output file name (boundary): {prefix}_boundary.vtu")
        return bnd

    log.info(f"Boundary file = {bnd} not found.")
    return ValueError()


def _check_for_file(file: Path | str, msg: str) -> Path | ValueError:
    file = Path(file)
    if not file.is_file():
        msg = msg.format(file=file)
        return ValueError(msg)
    return file


def _check_errors[T](var: T | ValueError, log: ILogger) -> T:
    match var:
        case ValueError() as e:
            log.error(e)
            raise ValueError from e
        case _:
            return var


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
) -> tuple[ProgramArgs, IIndexIterator]:
    """Process command line arguments raw into program structs."""
    log.info(*print_input_info(args))
    prefix = _get_prefix(args)
    input_dir, output_dir = _check_dirs_inputs(args)
    input_dir = _check_errors(input_dir, log)
    indexer = get_file_name_indexer(args.index, args.subindex, args.var, input_dir, log)
    ifirst = next(iter(indexer))
    """x: space, t: topology, b: boundary, u: displacement"""
    x, t, b, u = _get_mesh_names(args)
    space = _check_variable_format(x, ifirst, Path())
    disp = _check_variable_format(u, ifirst, Path())
    top = _check_for_file(t, r"Topology file = {file} not found.")
    bnd = _check_boundary_file(b, prefix, log)
    var = tuple([_check_variable_format(v, ifirst, input_dir) for v in args.var])
    if (
        _capture_err(space, log)
        and _capture_err(disp, log)
        and _capture_err(top, log)
        and _capture_err(bnd, log)
        and _capture_err_sequence(var, log)
    ):
        return ProgramArgs(
            prefix=prefix,
            input_folder=input_dir,
            output_folder=output_dir,
            time_series=args.time_series,
            progress_bar=args.progress_bar,
            binary=args.binary,
            compression=args.compression,
            cores=args.cores,
            tfile=top,
            bfile=bnd,
            space=space,
            disp=disp,
            var={v.name: v for v in var},
        ), indexer
    raise ValueError
