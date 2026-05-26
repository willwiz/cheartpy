import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io import chwrite_d_utf

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A2

    from cheartpy.mesh import CheartMesh


@dc.dataclass(slots=True)
class MergedMesh[F: np.floating, I: np.integer]:
    vol: CheartMesh[F, I]
    iface: CheartMesh[F, I]
    var: Mapping[str, A2[F]]

    def save(self, prefix: str, root: Path | None = None) -> None:
        root = root or Path.cwd()
        self.vol.save(root / f"{prefix}Planes")
        self.iface.save(root / f"{prefix}Interface")
        for k, v in self.var.items():
            chwrite_d_utf(root / f"{prefix}Planes{k}-0.D", v)
