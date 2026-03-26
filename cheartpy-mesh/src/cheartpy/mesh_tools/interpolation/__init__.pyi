import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
from pytools.arrays import A1, A2
from pytools.logging import ILogger

from cheartpy.mesh import CheartMesh

from ._parsing import InterpArgs, InterpKwargs

type INTERP_MAP[T: np.integer] = Mapping[int, A1[T]]

class _InterpolatKwargs(TypedDict, total=False):
    log: ILogger
    overwrite: bool

interp_parser: argparse.ArgumentParser

def create_quad_mesh_from_lin[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]: ...
def create_quad_mesh_from_lin_cylindrical[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]: ...
def get_interp_args(
    args: Sequence[str] | None = None,
) -> tuple[InterpArgs, InterpKwargs]: ...
def interp_var_l2q[T: np.floating, I: np.integer](l2qmap: INTERP_MAP[I], lin: A2[T]) -> A2[T]: ...
def interpolate_var_on_lin_topology[I: np.integer](
    l2qmap: INTERP_MAP[I], lin_var: Path, quad_var: Path, **kwargs: Unpack[_InterpolatKwargs]
) -> None: ...
def make_interp_api(args: InterpArgs, **kwargs: Unpack[InterpKwargs]) -> None: ...
def make_l2qmap[F: np.floating, I: np.integer](
    lin_mesh: CheartMesh[F, I], quad_mesh: CheartMesh[F, I]
) -> INTERP_MAP[I]: ...
