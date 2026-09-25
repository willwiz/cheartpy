import argparse

from cheartpy.search import AUTO

find_subparser = argparse.ArgumentParser("find", add_help=False)
_index_group = find_subparser.add_argument_group(title="Indexing")
_index_group.add_argument(
    "--index",
    "-i",
    nargs=3,
    dest="index",
    action="store",
    default=AUTO,
    type=int,
    metavar=("start", "end", "step"),
    help=(
        "Specify the start, end, and step for the range of data files. "
        "Will overwrite automated discovery."
    ),
)
_index_group_sub = _index_group.add_mutually_exclusive_group()
_index_group_sub.add_argument(
    "--subindex",
    "-si",
    dest="subindex",
    nargs=3,
    action="store",
    type=int,
    metavar=("start", "end", "step"),
    default=None,
    help=("Specify the start, end, and step for the range of subindex in the data files. "),
)
_index_group_sub.add_argument(
    "--subindex-auto",
    "-sa",
    action="store_const",
    dest="subindex",
    const=AUTO,
    help=(
        "Sub indices should be automatically determined based on the data in the input directory."
    ),
)
