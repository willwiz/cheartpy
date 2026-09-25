import argparse

index_subparser = argparse.ArgumentParser("index", add_help=False)
_index_group = index_subparser.add_argument_group(title="Indexing")
_index_group.add_argument(
    "--index",
    "-i",
    nargs=3,
    dest="index",
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
    required=True,
    help=("Specify the start, end, and step for the range of data files. "),
)
_index_group.add_argument(
    "--subindex",
    "-si",
    dest="subindex",
    nargs=3,
    action="store",
    type=int,
    metavar=("start", "end", "step"),
    default=None,
    help=("Specify the start, end, and step for the range of data files. "),
)
