import argparse

from pytools.logging.trait import LogLevel
from pytools.parsing import EnumGetter

setting_parser = argparse.ArgumentParser(add_help=False)
_settinggroup = setting_parser.add_argument_group(title="Settings")
_settinggroup.add_argument("--no-progressbar", action="store_false", dest="prog_bar")
_settinggroup.add_argument(
    "--log",
    type=EnumGetter(LogLevel, upper_case=True),
    choices=LogLevel._member_names_,
    default=LogLevel.INFO,
)
_settinggroup.add_argument("--binary", action="store_true")
_settinggroup.add_argument("--no-compression", dest="compress", action="store_false")
_settinggroup.add_argument("--n-cores", "-n", type=int, dest="cores")
