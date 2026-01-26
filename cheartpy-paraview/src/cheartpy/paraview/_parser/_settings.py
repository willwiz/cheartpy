import argparse

from pytools.logging import LogLevel
from pytools.parsing import EnumGetter

setting_parser = argparse.ArgumentParser(add_help=False)
_settinggroup = setting_parser.add_argument_group(title="Settings")
_settinggroup.add_argument(
    "--log",
    type=EnumGetter(LogLevel, upper_case=True),
    choices=LogLevel._member_names_,
    default=LogLevel.INFO,
)
_settinggroup.add_argument("--binary", action="store_true")
_settinggroup.add_argument("--no-progressbar", action="store_false", dest="prog_bar")
_settinggroup.add_argument("--no-compression", dest="compress", action="store_false")
multiprocessing_parser = argparse.ArgumentParser(add_help=False)
multiprocessing_group = multiprocessing_parser.add_argument_group(
    title="Multiprocessing (Choose 1)",
)
_mutually_exclusive_group = multiprocessing_group.add_mutually_exclusive_group(required=False)
_mutually_exclusive_group.add_argument("--core", type=int, dest="core", default=None)
_mutually_exclusive_group.add_argument("--thread", type=int, dest="thread", default=None)
_mutually_exclusive_group.add_argument("--interpreter", type=int, dest="interpreter", default=None)
