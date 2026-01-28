import dataclasses as dc
from pathlib import Path
from typing import Literal

from cheartpy.mesh.api import import_cheart_mesh
from cheartpy.search.api import get_var_index

from .interpolate.interpolation import interpolate_var_on_lin_topology, make_l2qmap
from .interpolate.parsing import interp_parser


@dc.dataclass(slots=True)
class InterpInputArgs:
    var: list[str]
    lin: str
    quad: str
    input_dir: Path
    suffix: str
    ext: Literal["D", "D.gz"]
    index: tuple[int, int, int] | None
    sub_index: tuple[int, int, int] | None
    sub_auto: bool


def main_interp(inp: InterpInputArgs) -> None:
    lin_mesh = import_cheart_mesh(inp.lin).unwrap()
    quad_mesh = import_cheart_mesh(inp.quad).unwrap()
    l2qmap = make_l2qmap(lin_mesh, quad_mesh)
    items = get_var_index(
        [v.name for v in Path(inp.input_dir).glob(f"{inp.var[0]}-*.{inp.ext}")],
        f"{inp.var[0]}",
    ).unwrap()
    for v in inp.var:
        for i in items:
            interpolate_var_on_lin_topology(
                l2qmap,
                inp.input_dir / f"{v}-{i}.{inp.ext}",
                inp.input_dir / f"{v}_{inp.suffix}-{i}.{inp.ext}",
            )


def main_cli(cmd_args: list[str] | None = None) -> None:
    args = interp_parser.parse_args(
        cmd_args,
        namespace=InterpInputArgs([], "", "", Path(), "Quad", "D", None, None, sub_auto=True),
    )
    main_interp(args)


if __name__ == "__main__":
    main_cli()
