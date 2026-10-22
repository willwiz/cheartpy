from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from pytools.arrays import A1


class AbaqusWriterKwargs(TypedDict, total=False):
    header: Sequence[str]
    elset: Mapping[str, A1[np.integer]]
    nset: Mapping[str, A1[np.integer]]
