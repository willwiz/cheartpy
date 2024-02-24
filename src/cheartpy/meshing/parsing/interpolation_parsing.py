import argparse


element_parser = argparse.ArgumentParser("elem", add_help=False)
element_parser.add_argument(
    "--elem", "-e", type=str.upper, default=None, choices=["HEX", "TET", "SQUARE"])

interp_parser = argparse.ArgumentParser(
    "interp", description="""interpolate data from linear topology to quadratic topology""", parents=[element_parser]
)
interp_parser.add_argument(
    "--topologies",
    "-t",
    type=str,
    nargs=2,
    metavar=("lin_top", "quad_top"),
    default=None,
    help="OPTIONAL: this tool will be set to make the make map mode. Requires the two topology to be supplied",
)
interp_parser.add_argument(
    "--use-map",
    type=str,
    default=None,
    help="OPTIONAL: used a premade map to save time",
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
    type=str,
    default="",
    help="OPTIONAL: specify a folder for the where the variables are stored. NOT YET IMPLEMENTED",
)
interp_parser.add_argument(
    "--suffix",
    "-s",
    dest="suffix",
    type=str,
    default="Quad",
    help="OPTIONAL: output file will have [tag] appended to the end of name before index numbers and extension",
)
interp_parser.add_argument(
    "vars",
    nargs="+",
    help="names to files/variables.",
    type=str,
)


map_parser = argparse.ArgumentParser(
    "make-map",
    description="""
        create a interpolation map in json format
    """,
    parents=[element_parser],
)
map_parser.add_argument(
    "--prefix",
    "-p",
    dest="prefix",
    type=str,
    default=None,
    help="OPTIONAL: output file will be the common prefix + _l2q.map",
)
map_parser.add_argument(
    "lin",
    help="linear topology file name",
    type=str,
)
map_parser.add_argument(
    "quad",
    help="quadratic topology file name",
    type=str,
)
