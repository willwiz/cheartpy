import dataclasses as dc

from cheartpy.cheart_core.pytools import join_fields
from .base_types.topologies import _CheartTopology
from .aliases import TopologyInterfaceType
from typing import TextIO


@dc.dataclass
class TopInterface:
    name: str
    method: TopologyInterfaceType
    topologies: list[_CheartTopology] = dc.field(default_factory=list)

    def write(self, f: TextIO):
        string = join_fields(self.method, *self.topologies)
        f.write(f"!DefInterface={{{string}}}\n")
