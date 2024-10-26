import os


def path(*v: str) -> str:
    return os.path.join(*v)
