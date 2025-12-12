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
    help=(
        "MANDATORY: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
_index_group.add_argument(
    "--subindex",
    "-si",
    dest="subindex",
    nargs=3,
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
    help=(
        "OPTIONAL: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
