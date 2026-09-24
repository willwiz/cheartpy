import argparse
from pathlib import Path

find_topology_parser = argparse.ArgumentParser(add_help=False)
_topology_group = find_topology_parser.add_argument_group(title="Topology")
_topology_group.add_argument(
    "--mesh",
    required=True,
    dest="mesh_or_top",
    action="store",
    type=Path,
    help="OPTIONAL: supply a prefix for the mesh files",
)
_topology_group.add_argument(
    "--space",
    "-x",
    dest="space",
    action="store",
    type=Path,
    default=None,
    help="OPTIONAL: supply a prefix for the space file(s)",
)
_topology_group.add_argument(
    "--disp",
    "-u",
    dest="disp",
    action="store",
    type=Path,
    default=None,
    help=(
        "OPTIONAL: supply a prefix for the disp file(s). Only include this if you want the space "
        "to be updated each time step."
    ),
)
_topology_group.add_argument(
    "--boundary",
    "-b",
    dest="boundary",
    action="store",
    type=Path,
    default=None,
    help=(
        "OPTIONAL: supply a relative path and file name from the current directory "
        "to the boundary file, the default is mesh_FE.B"
    ),
)


index_topology_parser = argparse.ArgumentParser(add_help=False)
_topology_group = index_topology_parser.add_argument_group(title="Topology")
_topology_group.add_argument(
    "--space",
    "-x",
    required=True,
    dest="space",
    action="store",
    type=str,
    help="OPTIONAL: supply a prefix for the mesh files",
)
_topology_group.add_argument(
    "--disp",
    "-u",
    dest="disp",
    action="store",
    type=Path,
    default=None,
    help=(
        "OPTIONAL: supply a prefix for the disp file(s). Only include this if you want the space "
        "to be updated each time step."
    ),
)
_topology_group.add_argument(
    "--top",
    "-t",
    required=True,
    dest="mesh_or_top",
    action="store",
    type=Path,
    help=(
        "MANDATORY: supply a relative path and file name from the current directory "
        "to the topology file, the default is mesh_FE.T"
    ),
)
_topology_group.add_argument(
    "--boundary",
    "-b",
    dest="boundary",
    action="store",
    type=Path,
    default=None,
    help=(
        "OPTIONAL: supply a relative path and file name from the current directory "
        "to the boundary file, the default is mesh_FE.B"
    ),
)
