import dataclasses as dc
from glob import glob
from typing import Literal
from ..io.indexing.search import get_var_index
from ..cheart_mesh.api import import_cheart_mesh
from .interpolate.interpolation import interpolate_var_on_lin_topology, make_l2qmap
from .interpolate.parsing import interp_parser
import argparse


@dc.dataclass(slots=True)
class InterpInputArgs:
    vars: list[str]
    lin_mesh: str
    quad_mesh: str
    input_folder: str = ""
    suffix: str = "Quad"
    sfx: Literal["D", "D.gz"] = "D"
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


def main_interp(inp: InterpInputArgs):
    lin_mesh = import_cheart_mesh(inp.lin_mesh)
    quad_mesh = import_cheart_mesh(inp.quad_mesh)
    L2Q = make_l2qmap(lin_mesh, quad_mesh)
    items = get_var_index(
        glob(f"{inp.vars[0]}-*.{inp.suffix}", root_dir=inp.input_folder),
        f"{inp.vars[0]}",
    )
    for v in inp.vars:
        for i in items:
            interpolate_var_on_lin_topology(
                L2Q,
                f"{inp.input_folder}/{v}-{i}.{inp.sfx}",
                f"{inp.input_folder}/{v}_{inp.suffix}-{i}.{inp.sfx}",
            )


def main_cli(cmd_args: list[str] | None = None):
    args = interp_parser.parse_args(cmd_args)
    inp = check_args_interp(args)
    main_interp(inp)


if __name__ == "__main__":
    main_cli()
