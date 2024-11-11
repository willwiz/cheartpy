import dataclasses as dc
from .interpolate.parsing import interp_parser
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


def main_interp(inp: InterpInputArgs):
    # lin_mesh = import_cheart_mesh(inp.lin_mesh)
    # quad_mesh = import_cheart_mesh(inp.quad_mesh)
    # L2Q = make_l2qmap(lin_mesh, quad_mesh)
    ...


def main_cli(cmd_args: list[str] | None = None):
    args = interp_parser.parse_args(cmd_args)
    inp = check_args_interp(args)
    main_interp(inp)


if __name__ == "__main__":
    main_cli()
