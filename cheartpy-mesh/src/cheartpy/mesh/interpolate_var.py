import argparse
import dataclasses as dc
from pathlib import Path
from typing import Literal

from cheartpy.search.api import get_var_index

from .api import import_cheart_mesh
from .interpolate.interpolation import interpolate_var_on_lin_topology, make_l2qmap
from .interpolate.parsing import interp_parser


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


def main_interp(inp: InterpInputArgs) -> None:
    lin_mesh = import_cheart_mesh(inp.lin_mesh)
    quad_mesh = import_cheart_mesh(inp.quad_mesh)
    l2qmap = make_l2qmap(lin_mesh, quad_mesh)
    items = get_var_index(
        [v.name for v in Path(inp.input_folder).glob(f"{inp.vars[0]}-*.{inp.suffix}")],
        f"{inp.vars[0]}",
    )
    for v in inp.vars:
        for i in items:
            interpolate_var_on_lin_topology(
                l2qmap,
                f"{inp.input_folder}/{v}-{i}.{inp.sfx}",
                f"{inp.input_folder}/{v}_{inp.suffix}-{i}.{inp.sfx}",
            )


def main_cli(cmd_args: list[str] | None = None) -> None:
    args = interp_parser.parse_args(cmd_args)
    inp = check_args_interp(args)
    main_interp(inp)


if __name__ == "__main__":
    main_cli()
