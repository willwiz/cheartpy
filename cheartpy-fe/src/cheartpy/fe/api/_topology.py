from pathlib import Path
from typing import TYPE_CHECKING, TypeIs

from cheartpy.fe.aliases import CheartTopInterfaceType, VariableExportEnum, VariableExportFormat
from cheartpy.fe.impl import (
    CheartTopology,
    ManyToOneTopInterface,
    NullTopology,
    OneToOneTopInterface,
)
from cheartpy.fe.utils import get_enum

from ._basis import create_basis, create_boundary_basis

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cheartpy.fe.aliases._topology import EmbbededTopologyDef, TopologyDef, VolumeTopologyDef
    from cheartpy.fe.trait import ICheartBasis, ICheartTopology, ITopInterface

def _hash_tops(tops: list[ICheartTopology] | list[str]) -> str:
    names = [str(t) for t in tops]
    return "_".join(names)


# @overload
# def create_topology(
#     name: str, basis: ICheartBasis, mesh: Path | str, fmt: VariableExportFormat = ...
# ) -> CheartTopology: ...
# @overload
# def create_topology(
#     name: str, basis: None, mesh: Path | str, fmt: VariableExportFormat = ...
# ) -> NullTopology: ...
def create_topology(
    name: str,
    basis: ICheartBasis | None,
    mesh: Path | str,
    fmt: VariableExportFormat = "TXT",
) -> CheartTopology | NullTopology:
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


# @overload
# def create_top_interface(
#     method: Literal["OneToOne"],
#     topologies: list[ICheartTopology],
#     *,
#     nest_in_bnd: int | None = None,
# ) -> OneToOneTopInterface: ...
# @overload
# def create_top_interface(
#     method: Literal["ManyToOne"],
#     topologies: list[ICheartTopology],
#     *,
#     master: ICheartTopology,
#     interface_file: Path | str,
#     nest_in_bnd: int | None = None,
# ) -> ManyToOneTopInterface: ...
def create_top_interface(
    method: CheartTopInterfaceType,
    topologies: list[ICheartTopology],
    *,
    master: ICheartTopology | None = None,
    interface_file: Path | str | None = None,
    nest_in_bnd: int | None = None,
) -> CheartTopInterfaceType:
    match method:
        case "OneToOne":
            name = _hash_tops(topologies)
            return OneToOneTopInterface(name, topologies)
        case "ManyToOne":
            if master is None:
                msg = "ManyToOne requires a master_topology"
                raise ValueError(msg)
            if interface_file is None:
                msg = "ManyToOne requires a interface_file"
                raise ValueError(msg)
            name = _hash_tops(topologies) + ":" + str(master)
            return ManyToOneTopInterface(
                name,
                topologies,
                master,
                Path(interface_file),
                nest_in_bnd,
            )


def is_volume_topology[T](defn: TopologyDef[T]) -> TypeIs[VolumeTopologyDef]:
    return "elem" in defn


def is_embbeded_topology[T](defn: TopologyDef[T]) -> TypeIs[EmbbededTopologyDef[T]]:
    return "master" in defn


def create_volume_topologies[T](
    dct: Mapping[T, VolumeTopologyDef],
) -> tuple[Mapping[T, ICheartTopology], Sequence[ITopInterface]]:
    """Return dict of topologies and sequence of interfaces.

    Convenient way of creating topology basis and interfaces from a dictionary of definitions.

    Parameters
    ----------
    dct
        Mapping of topology name to mesh prefix.

    Returns
    -------
    tuple[Mapping[T, ICheartTopology], Sequence[ITopInterface]]
        Mapping of topology name to topology and sequence of interfaces.

    """
    basis: Mapping[T, ICheartBasis | None] = {
        k: create_basis(elem, "NL", v["order"]) if (elem := v["elem"]) else None
        for k, v in dct.items()
    }
    tops: Mapping[T, ICheartTopology] = {
        k: create_topology(f"TP{k}", basis=basis[k], mesh=v["mesh"]) for k, v in dct.items()
    }
    interface = create_top_interface("OneToOne", [*tops.values()])
    return tops, [interface]


def create_embbeded_topologies[T](
    volumes: Mapping[T, ICheartTopology],
    dct: Mapping[T, EmbbededTopologyDef[T]],
) -> tuple[Mapping[T, ICheartTopology], Sequence[ITopInterface]]:

    tops = {
        k: create_topology(
            f"TP{k}", basis=create_boundary_basis(volumes[v["master"]].get_basis()), mesh=v["mesh"]
        )
        for k, v in dct.items()
    }
    for k, v in dct.items():
        tops[k].create_in_boundary(volumes[v["master"]], v["bnd"])
    interfaces = [
        create_top_interface(
            "ManyToOne",
            [t],
            master=volumes[dct[k]["master"]],
            interface_file=dct[k]["mesh"].parent / f"iface-{k}.IN",
            nest_in_bnd=dct[k]["bnd"],
        )
        for k, t in tops.items()
    ]
    return {**tops}, interfaces


def create_topologies[T](
    dct: Mapping[T, TopologyDef[T]],
) -> tuple[Mapping[T, ICheartTopology], Sequence[ITopInterface]]:
    """Return dict of topologies and sequence of interfaces.

    Convenient way of creating topology basis and interfaces from a dictionary of definitions.

    Parameters
    ----------
    volumes
        Mapping of topology name to volume topology.
    dct
        Mapping of topology name to mesh prefix, master topology and boundary tag.

    Returns
    -------
    tuple[Mapping[T, ICheartTopology], Sequence[ITopInterface]]
        Mapping of topology name to topology and sequence of interfaces.

    """
    volumes = {k: v for k, v in dct.items() if is_volume_topology(v)}
    embbeded = {k: v for k, v in dct.items() if is_embbeded_topology(v)}
    tops, interfaces = create_volume_topologies(volumes)
    embbeded_tops, embbeded_interfaces = create_embbeded_topologies(tops, embbeded)
    return {**tops, **embbeded_tops}, [*interfaces, *embbeded_interfaces]
