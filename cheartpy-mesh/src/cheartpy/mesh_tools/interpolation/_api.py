import re
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import numpy as np
from cheartpy.search import get_var_index
from pytools.logging import get_logger
from pytools.parallel import ThreadedRunner
from pytools.progress import ProgressBar
from pytools.result import Err, Ok, Result

from cheartpy.mesh import CheartMesh, import_cheart_mesh

from ._interpolation import export_quad_var_from_lin, make_l2qmap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._parsing import InterpArgs, InterpKwargs


def interp_vars_api[F: np.floating, I: np.integer](
    lin: CheartMesh[F, I],
    quad: CheartMesh[F, I],
    vs: Mapping[str, str],
    **kwargs: Unpack[InterpKwargs],
) -> None:
    logger = get_logger()
    input_dir = kwargs.get("input_dir") or Path.cwd()
    ext = kwargs.get("ext") or "D"
    l2qmap = make_l2qmap(lin, quad)
    arg_list = {
        k: (v, get_var_index([f.name for f in input_dir.glob(f"{k}-*.{ext}")], k).unwrap())
        for k, v in vs.items()
    }
    bart = ProgressBar(sum(len(items) for _, items in arg_list.values()))
    with ThreadedRunner(thread=(kwargs.get("threads") or 1), prog_bar=bart) as runner:
        for k, (v, items) in arg_list.items():
            for i in items:
                runner.submit(
                    export_quad_var_from_lin,
                    l2qmap,
                    input_dir / f"{k}-{i}.{ext}",
                    input_dir / f"{v}-{i}.{ext}",
                    log=logger,
                    overwrite=kwargs.get("overwrite", False),
                )


POSTFIX = re.compile(r"\[(?P<postfix>\w+(?:\s*,\s*\w+)*)\]")


def parse_postfix(args: Sequence[str], postfix: Sequence[str] | None) -> Result[Mapping[str, str]]:
    if postfix is None:
        return Ok({a: f"{a}_Quad" for a in args})
    if len(postfix) != len(args):
        msg = (
            "Number of postfixes does not match number of variables\n"
            f"Postfixes: {postfix}\n"
            f"Variables: {args}"
        )
        return Err(ValueError(msg))
    return Ok({a: f"{a}{p}" for a, p in zip(args, postfix, strict=True)})


def make_interp_cli2(args: InterpArgs, **kwargs: Unpack[InterpKwargs]) -> None:
    lin_mesh = import_cheart_mesh(args["lin"]).unwrap()
    quad_mesh = import_cheart_mesh(args["quad"]).unwrap()

    interp_vars_api(
        lin_mesh,
        quad_mesh,
        {k: f"{k}{kwargs.get('suffix', 'Quad')}" for k in args["vars"]},
        **kwargs,
    )


def make_interp_cli(args: InterpArgs, **kwargs: Unpack[InterpKwargs]) -> None:
    lin_mesh = import_cheart_mesh(args["lin"]).unwrap()
    quad_mesh = import_cheart_mesh(args["quad"]).unwrap()
    arg_list = parse_postfix(args["vars"], args.get("postfix")).unwrap()
    interp_vars_api(lin_mesh, quad_mesh, arg_list, **kwargs)
