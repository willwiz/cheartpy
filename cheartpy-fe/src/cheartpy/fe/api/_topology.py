from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from cheartpy.fe.aliases import (
    CheartTopInterfaceType,
    VariableExportEnum,
    VariableExportFormat,
)
from cheartpy.fe.impl import (
    CheartTopology,
    ManyToOneTopInterface,
    NullTopology,
    OneToOneTopInterface,
)
from cheartpy.fe.string_tools import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import (
        ICheartBasis,
        ICheartTopology,
        ITopInterface,
    )


def hash_tops(tops: list[ICheartTopology] | list[str]) -> str:
    names = [str(t) for t in tops]
    return "_".join(names)


@overload
def create_topology(
    name: str, basis: ICheartBasis, mesh: Path | str, fmt: VariableExportFormat = ...
) -> CheartTopology: ...
@overload
def create_topology(
    name: str, basis: None, mesh: Path | str, fmt: VariableExportFormat = ...
) -> NullTopology: ...
def create_topology(
    name: str,
    basis: ICheartBasis | None,
    mesh: Path | str,
    fmt: VariableExportFormat = "TXT",
) -> ICheartTopology:
    if basis is None:
        return NullTopology()
    _fmt = get_enum(fmt, VariableExportEnum)
    return CheartTopology(name, basis, Path(mesh), _fmt)


def create_embedded_topology(
    name: str,
    embedded_top: ICheartTopology,
    mesh: Path | str,
    fmt: VariableExportFormat = "TXT",
) -> CheartTopology:
    _fmt = get_enum(fmt, VariableExportEnum)
    return CheartTopology(name, None, Path(mesh), _fmt, embedded=embedded_top)


@overload
def create_top_interface(
    method: Literal["OneToOne"],
    topologies: list[ICheartTopology],
    *,
    nest_in_bnd: int | None = None,
) -> OneToOneTopInterface: ...
@overload
def create_top_interface(
    method: Literal["ManyToOne"],
    topologies: list[ICheartTopology],
    *,
    master: ICheartTopology,
    interface_file: Path | str,
    nest_in_bnd: int | None = None,
) -> ManyToOneTopInterface: ...
def create_top_interface(
    method: CheartTopInterfaceType,
    topologies: list[ICheartTopology],
    *,
    master: ICheartTopology | None = None,
    interface_file: Path | str | None = None,
    nest_in_bnd: int | None = None,
) -> ITopInterface:
    match method:
        case "OneToOne":
            name = hash_tops(topologies)
            return OneToOneTopInterface(name, topologies)
        case "ManyToOne":
            if master is None:
                msg = "ManyToOne requires a master_topology"
                raise ValueError(msg)
            if interface_file is None:
                msg = "ManyToOne requires a interface_file"
                raise ValueError(msg)
            name = hash_tops(topologies) + ":" + str(master)
            return ManyToOneTopInterface(
                name,
                topologies,
                master,
                Path(interface_file),
                nest_in_bnd,
            )
