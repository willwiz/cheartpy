from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from cheartpy.search import get_var_index

from cheartpy.mesh import import_cheart_mesh

from ._interpolation import interpolate_var_on_lin_topology, make_l2qmap

if TYPE_CHECKING:
    from ._parsing import InterpArgs, InterpKwargs


def make_interp_api(args: InterpArgs, **kwargs: Unpack[InterpKwargs]) -> None:
    input_dir = kwargs.get("input_dir", Path.cwd())
    ext = kwargs.get("ext", "D")
    suffix = kwargs.get("suffix", "Quad")
    lin_mesh = import_cheart_mesh(args["lin"]).unwrap()
    quad_mesh = import_cheart_mesh(args["quad"]).unwrap()
    l2qmap = make_l2qmap(lin_mesh, quad_mesh)
    items = get_var_index(
        [v.name for v in input_dir.glob(f"{args['vars'][0]}-*.{ext}")],
        f"{args['vars'][0]}",
    ).unwrap()
    for v in args["vars"]:
        for i in items:
            interpolate_var_on_lin_topology(
                l2qmap,
                input_dir / f"{v}-{i}.{ext}",
                input_dir / f"{v}_{suffix}-{i}.{ext}",
            )
