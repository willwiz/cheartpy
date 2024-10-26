from typing import Literal
import dataclasses as dc
from cheartpy.meshing.core.interpolation_core import (
    string_head,
    gen_map,
)
from cheartpy.meshing.parsing.interpolation_parsing import map_parser
from cheartpy.meshing.cheart.io import CHRead_header_utf
from cheartpy.tools.progress_bar import ProgressBar
import json
import argparse


@dc.dataclass(slots=True)
class MapInputArgs:
    prefix: str
    lin: str
    quad: str
    kind: Literal["TET", "HEX", "SQUARE"] | None


def check_args_map(args: argparse.Namespace) -> MapInputArgs:
    if args.prefix is None:
        prefix = (lambda x: "l2q.map" if x == "" else x + "_l2q.map")(
            string_head(args.lin, args.quad).rstrip(" .-_")
        )
    else:
        prefix = args.prefix
    return MapInputArgs(prefix, args.lin, args.quad, args.elem)


def make_map(lin: str, quad: str, kind: Literal["TET", "HEX", "SQUARE"] | None):
    ne, nn = CHRead_header_utf(quad)
    print(f"Generating Map from {lin} to {quad}:")
    bar = ProgressBar(max=ne, prefix=">>Progress:")
    map = gen_map(lin, quad, kind, bar=bar.next)
    assert (
        len(map) == nn
    ), f"Mismatch in the size of map {
        len(map)} and number of nodes in the header {nn}"
    return map


def main_map(inp: MapInputArgs):
    top_map = make_map(inp.lin, inp.quad, inp.kind)
    with open(inp.prefix, "w") as fp:
        json.dump(top_map, fp)


def main_cli(cmd_args: list[str] | None = None):
    args = map_parser.parse_args(cmd_args)
    inp = check_args_map(args)
    main_map(inp)


if __name__ == "__main__":
    main_cli()
