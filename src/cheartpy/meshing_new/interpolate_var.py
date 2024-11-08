import os
import dataclasses as dc

from ..cheart_mesh.api import import_cheart_mesh
from ..meshing.core.interpolation_core import (
    split_Dfile_name,
    gen_map,
    load_map,
    intep_lin_to_quad,
    get_file_name_indexer,
)
from ..cheart_mesh.io import CHRead_d, CHWrite_d_utf
from ..tools.progress_bar import ProgressBar
from ..var_types import Mat, f64
from .interpolate.parsing import interp_parser
from .interpolate.interpolation import make_l2qmap
import argparse


@dc.dataclass(slots=True)
class InterpInputArgs:
    vars: list[str]
    lin_mesh: str
    quad_mesh: str
    suffix: str
    input_folder: str = ""
    index: tuple[int, int, int] | None = None
    sub_index: tuple[int, int, int] | None = None
    sub_auto: bool = False


def check_args_interp(args: argparse.Namespace) -> InterpInputArgs:
    return InterpInputArgs(
        args.vars,
        args.lin_mesh,
        args.quad_mesh,
        args.suffix,
        args.folder,
        args.index,
        args.sub_index,
        args.sub_auto,
    )


def map_vals(map: dict[int, list[int]], lin: Mat[f64], prefix: str):
    quad_data = intep_lin_to_quad(map, lin)
    CHWrite_d_utf(prefix, quad_data)


def main_interp_file(file: str, map: dict[int, list[int]], suffix: str):
    prefix = split_Dfile_name(file, suffix)
    lin_data = CHRead_d(file)
    map_vals(map, lin_data, prefix)


def main_interp_var(var: str, map: dict[int, list[int]], inp: InterpInputArgs):
    indexer = get_file_name_indexer(
        [var], inp.index, inp.sub_auto, inp.sub_index, inp.input_folder
    )
    bart = ProgressBar(indexer.size, "Interp")
    if indexer.size == 0:
        raise ValueError(f"No files found for variable {var}")
    for i in indexer.get_generator():
        input_file = os.path.join(inp.input_folder, f"{var}-{i}.D")
        output_file = os.path.join(inp.input_folder, f"{var}{inp.suffix}-{i}.D")
        lin_data = CHRead_d(input_file)
        map_vals(map, lin_data, output_file)
        bart.next()


def main_interp(inp: InterpInputArgs):
    lin_mesh = import_cheart_mesh(inp.lin_mesh)
    quad_mesh = import_cheart_mesh(inp.quad_mesh)
    L2Q = make_l2qmap(lin_mesh, quad_mesh)


def main_cli(cmd_args: list[str] | None = None):
    args = interp_parser.parse_args(cmd_args)
    inp = check_args_interp(args)
    main_interp(inp)


if __name__ == "__main__":
    main_cli()
