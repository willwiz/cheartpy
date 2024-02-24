from cheartpy.meshing.parsing.interpolation_parsing import map_parser, interp_parser
from cheartpy.meshing.cheart_data_interpolation import check_args_interp, main_interp
from cheartpy.meshing.cheart_topology_map import check_args_map, main_map
import argparse

main_parser = argparse.ArgumentParser(
    prog="interpDfiles",
)
subparsers = main_parser.add_subparsers(
    help="Collective of subprogram", dest="cmd")
subparsers.add_parser("make-map", description="call make map program",
                      add_help=False, parents=[map_parser])
subparsers.add_parser("interp", description="call make interpolate program",
                      add_help=False, parents=[interp_parser])


def main_cli(cmd_args: list[str] | None = None):
    args = main_parser.parse_args(cmd_args)
    match args.cmd:
        case "make-map":
            inp = check_args_map(args)
            main_map(inp)
        case "interp":
            inp = check_args_interp(args)
            main_interp(inp)
        case _:
            raise ValueError(f"subparser {args.cmd} is not implemented")


if __name__ == "__main__":
    main_cli()
