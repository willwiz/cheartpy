import os
from glob import glob
from .basiclogging import _Logger, NullLogger


def path(*v: str) -> str:
    """
    Joins args as a system path str
    """
    return os.path.join(*v)


def Clear_Dir(folder: str, LOG: _Logger = NullLogger()) -> None:
    """
    Remove all files in directory
    """
    if not os.path.isdir(folder):
        LOG.warn(f"Dir {folder} was not found.")
        return
    [os.remove(s) for s in glob(f"{folder}/*") if os.path.isfile(s)]
