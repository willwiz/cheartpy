import argparse

from cheartpy.search.trait import AUTO

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
        "MANDATORY: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
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
    help=(
        "OPTIONAL: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
_index_group_sub.add_argument(
    "--subindex-auto",
    action="store_const",
    dest="subindex",
    const=AUTO,
    help=(
        "OPTIONAL: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
