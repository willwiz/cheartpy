import os
from typing import Literal
import dataclasses as dc
from cheartpy.meshing.core.interpolation_core import (
    split_Dfile_name,
    gen_map,
    load_map,
    intep_lin_to_quad,
    get_file_name_indexer,
)
from cheartpy.meshing.parsing.interpolation_parsing import interp_parser
from cheartpy.io.cheartio import CHRead_d, CHWrite_d_utf
from cheartpy.tools.progress_bar import ProgressBar
from cheartpy.types import Mat, f64
import argparse


@dc.dataclass(slots=True)
class InterpInputArgs:
    vars: list[str]
    suffix: str
    kind: Literal["TET", "HEX", "SQUARE"] | None
    input_folder: str = ""
    index: tuple[int, int, int] | None = None
    sub_index: tuple[int, int, int] | None = None
    sub_auto: bool = False
    use_map: str | None = None
    topologies: tuple[str, str] | None = None


def check_args_interp(args: argparse.Namespace) -> InterpInputArgs:
    return InterpInputArgs(
        args.vars, args.suffix, args.elem, args.folder, args.index, args.sub_index, args.sub_auto, args.use_map
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
        [var], inp.index, inp.sub_auto, inp.sub_index, inp.input_folder)
    bart = ProgressBar(indexer.size, "Interp")
    if indexer.size == 0:
        raise ValueError(f"No files found for variable {var}")
    for i in indexer.get_generator():
        input_file = os.path.join(inp.input_folder, f"{var}-{i}.D")
        output_file = os.path.join(
            inp.input_folder, f"{var}{inp.suffix}-{i}.D")
        lin_data = CHRead_d(input_file)
        map_vals(map, lin_data, output_file)
        bart.next()


def main_interp(inp: InterpInputArgs):
    if inp.use_map is not None:
        map = load_map(inp.use_map)
    elif inp.topologies is not None:
        map = gen_map(inp.topologies[0], inp.topologies[1], inp.kind)
    else:
        raise ValueError(
            f"Either supply the lin to quad mapping or supply the topologies")
    for v in inp.vars:
        if os.path.isfile(v):
            main_interp_file(v, map, inp.suffix)
        elif os.path.isfile(os.path.join(inp.input_folder, v)):
            main_interp_file(os.path.join(
                inp.input_folder, v), map, inp.suffix)
        else:
            main_interp_var(v, map, inp)


def main_cli(cmd_args: list[str] | None = None):
    args = interp_parser.parse_args(cmd_args)
    inp = check_args_interp(args)
    main_interp(inp)


if __name__ == "__main__":
    main_cli()
