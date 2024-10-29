__all__ = ["check_for_meshes"]
import os


def check_for_meshes(*names: str) -> bool:
    meshes = [w for name in names for w in [f"{name}_FE.{s}" for s in ["X", "T", "B"]]]
    return all(os.path.isfile(s) for s in meshes)
