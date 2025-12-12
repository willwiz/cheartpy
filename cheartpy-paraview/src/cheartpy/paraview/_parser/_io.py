import argparse
from pathlib import Path

io_parser = argparse.ArgumentParser(add_help=False)
_iogroup = io_parser.add_argument_group(title="IO")
_iogroup.add_argument(
    "--prefix",
    "-p",
    type=str,
    help=(
        "OPTIONAL: supply a prefix name to be used for the exported vtu files."
        'If -p is not supplied, then "paraview" will be uses, e.g., paraview-#.D'
    ),
)
_iogroup.add_argument(
    "--folder",
    "-f",
    dest="input_dir",
    default=Path(),
    action="store",
    type=Path,
    help=(
        "OPTIONAL: supply the path to the folder where the .D files are stored. "
        "If -f is not supplied, then the path is assumed to be the current folder."
    ),
)
_iogroup.add_argument(
    "--out-folder",
    "-o",
    dest="output_dir",
    action="store",
    type=str,
    help=(
        "OPTIONAL: supply the path to the folder where the vtu outputs should be saved to. "
        "If -f is not supplied, then the path is assumed to be the current folder."
    ),
)
