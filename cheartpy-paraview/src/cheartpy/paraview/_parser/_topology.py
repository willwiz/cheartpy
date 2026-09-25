import argparse
from pathlib import Path

find_topology_parser = argparse.ArgumentParser(add_help=False)
_topology_group = find_topology_parser.add_argument_group(title="Topology")
_topology_group.add_argument(
    "--mesh",
    "-m",
    required=True,
    dest="mesh",
    action="store",
    type=Path,
    help="Path with a prefix for the mesh files, e.g., mesh/mesh_FE.",
)
_topology_group.add_argument(
    "--space",
    "-x",
    dest="space",
    action="store",
    type=Path,
    default=None,
    help=(
        "Override automatically discovered space file prefix.\n"
        "- File: all vtu will use"
        "- Path with prefix: vtu will look for files of the form parent/name-{i}.D to import"
        "- prefix: vtu will look for files of the form prefix-{i}.D in the input directory"
    ),
)
_topology_group.add_argument(
    "--disp",
    "-u",
    dest="disp",
    action="store",
    type=Path,
    default=None,
    help=("Prefix for files to update the space each time step."),
)
_topology_group.add_argument(
    "--boundary",
    "-b",
    dest="boundary",
    action="store",
    type=Path,
    default=None,
    help=("Override automatically discovered boundary file prefix."),
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
    help="Path to the space file, e.g., mesh_FE.X or a prefix to import prefix-{i}.D",
)
_topology_group.add_argument(
    "--disp",
    "-u",
    dest="disp",
    action="store",
    type=Path,
    default=None,
    help=("Prefix for files to update the space each time step, e.g., disp-{i}.D. "),
)
_topology_group.add_argument(
    "--top",
    "-t",
    required=True,
    dest="top",
    action="store",
    type=Path,
    help=("Path to the topology file, e.g., mesh_FE.T."),
)
_topology_group.add_argument(
    "--boundary",
    "-b",
    dest="boundary",
    action="store",
    type=Path,
    default=None,
    help=(
        "Path to the boundary file, e.g., mesh_FE.B. If not given boundary will not be exported."
    ),
)
