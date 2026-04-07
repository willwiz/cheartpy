from ._api import parse_pfile_api
from ._argparse import parse_args


def parse_pfile_cli(args: list[str] | None = None) -> None:
    args, kwargs = parse_args(args).unwrap()
    for file in args["file"]:
        parse_pfile_api(file=file, **kwargs)
