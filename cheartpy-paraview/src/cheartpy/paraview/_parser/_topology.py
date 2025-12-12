import argparse
from pathlib import Path

find_topology_parser = argparse.ArgumentParser(add_help=False)
_topology_group = find_topology_parser.add_argument_group(title="Topology")
_topology_group.add_argument(
    "--mesh",
    required=True,
    dest="mesh_or_top",
    action="store",
    default="mesh",
    type=str,
    help="OPTIONAL: supply a prefix for the mesh files",
)
_topology_group.add_argument(
    "--space",
    "-x",
    dest="space",
    action="store",
    default=None,
    type=str,
    help="OPTIONAL: supply a prefix for the mesh files",
)
_topology_group.add_argument(
    "--boundary",
    "-b",
    dest="boundary",
    action="store",
    default=None,
    type=Path,
    help=(
        "MANDATORY: supply a relative path and file name from the current directory "
        "to the topology file, the default is mesh_FE.T"
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
    default=None,
    type=str,
    help="OPTIONAL: supply a prefix for the mesh files",
)
_topology_group.add_argument(
    "--top",
    "-t",
    required=True,
    dest="mesh_or_top",
    action="store",
    default=None,
    type=str,
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
    default=None,
    type=Path,
    help=(
        "MANDATORY: supply a relative path and file name from the current directory "
        "to the topology file, the default is mesh_FE.T"
    ),
)
