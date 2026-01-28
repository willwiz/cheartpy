import argparse
from pathlib import Path

interp_parser = argparse.ArgumentParser(
    "interp",
    description="""interpolate data from linear topology to quadratic topology""",
)
interp_parser.add_argument(
    "--index",
    "-i",
    nargs=3,
    type=int,
    default=None,
    metavar=("start", "end", "step"),
    help="OPTIONAL: specify the start, end, and step for the range of data files.",
)
interp_parser.add_argument(
    "--sub-index",
    nargs=3,
    type=int,
    default=None,
    metavar=("start", "end", "step"),
    help="OPTIONAL: specify subindices",
)
interp_parser.add_argument(
    "--sub-auto",
    action="store_true",
    help="OPTIONAL: automatically detect subindex",
)
interp_parser.add_argument(
    "--folder",
    "-f",
    type=Path,
    default=Path(),
    dest="input_dir",
    help="OPTIONAL: specify a folder for the where the variables are stored. NOT YET IMPLEMENTED",
)
interp_parser.add_argument(
    "--suffix",
    "-s",
    dest="suffix",
    type=str,
    default="Quad",
    help="OPTIONAL: output file will have [tag] appended before index numbers and extension",
)
interp_parser.add_argument(
    "--sfx",
    "-s",
    dest="sfx",
    choices=["D", "D.gz"],
    default="D",
    help="OPTIONAL: D file suffix",
)
interp_parser.add_argument(
    "--lin",
    "-l",
    type=str,
    required=True,
    help="file path to linear mesh",
)
interp_parser.add_argument(
    "--quad",
    "-q",
    type=str,
    required=True,
    help="file path to quadratic mesh",
)
interp_parser.add_argument(
    "vars",
    nargs="+",
    help="names to files/variables.",
    type=str,
)
