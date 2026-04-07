from ._api import parse_pfile_api
from ._argparse import parse_args


def parse_pfile_cli(args: list[str] | None = None) -> None:
    _args, _kwargs = parse_args(args).unwrap()
    for file in _args["file"]:
        parse_pfile_api(file=file, **_kwargs)
