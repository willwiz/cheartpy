import argparse

from cheartpy.mesh import import_cheart_mesh
from pytools.logging import get_logger

from .__logging__ import compose_header, format_input_kwargs, header_guard
from ._api import create_cheartmesh_from_abaqus_api
from .parsing import parse_cmdline_args
from .writer import AbaqusWriterKwargs, import_region_mask, write_inp_from_cheart
from .writer import get_cmdline_args as get_writer_cmdline_args

main_parser = argparse.ArgumentParser()
sub_parsers = main_parser.add_subparsers(dest="command")
sub_parsers.add_parser("import", add_help=False)
sub_parsers.add_parser("export", add_help=False)


def import_cli(cmd_args: list[str] | None = None) -> None:
    args, kwargs = parse_cmdline_args(args=cmd_args)
    log = get_logger(level=kwargs.get("log_level"))
    log.info(*compose_header(), *format_input_kwargs(args["files"], **kwargs))
    create_cheartmesh_from_abaqus_api(args["files"], **kwargs).unwrap()
    log.info(header_guard(" COMPLETE! "), "")


def export_cli(cmd_args: list[str] | None = None) -> None:
    args = get_writer_cmdline_args(cmd_args).unwrap()
    regions = import_region_mask(args.region_mask) if args.region_mask else None
    output_file = args.prefix or args.mesh
    output_file = output_file.with_suffix(".inp")
    mesh = import_cheart_mesh(args.mesh).unwrap()
    kwargs: AbaqusWriterKwargs = {"elset": regions} if regions else {}
    write_inp_from_cheart(mesh, output_file, **kwargs).unwrap()


def main(cmdline: list[str] | None = None) -> None:
    cmd, args = main_parser.parse_known_args(cmdline)
    match cmd.command:
        case "import":
            import_cli(args)
        case "export":
            export_cli(args)
        case _:
            main_parser.print_help()


if __name__ == "__main__":
    main()
