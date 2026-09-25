import argparse
from typing import get_args

from pytools.logging import LogLevel

setting_parser = argparse.ArgumentParser(add_help=False)
_settinggroup = setting_parser.add_argument_group(title="Settings")
_settinggroup.add_argument(
    "--log", type=str.upper, choices=get_args(LogLevel.__value__), default="INFO"
)
_settinggroup.add_argument("--binary", action="store_true", help="imported data is binary")
_settinggroup.add_argument("--no-progressbar", action="store_false", dest="prog_bar")
_settinggroup.add_argument("--no-compression", dest="compress", action="store_false")
multiprocessing_parser = argparse.ArgumentParser(add_help=False)
multiprocessing_group = multiprocessing_parser.add_argument_group(
    title="Multiprocessing (Choose 1)"
)
_mutually_exclusive_group = multiprocessing_group.add_mutually_exclusive_group(required=False)
_mutually_exclusive_group.add_argument("--core", type=int, dest="core", default=None)
_mutually_exclusive_group.add_argument("--thread", type=int, dest="thread", default=None)
_mutually_exclusive_group.add_argument("--interpreter", type=int, dest="interpreter", default=None)
