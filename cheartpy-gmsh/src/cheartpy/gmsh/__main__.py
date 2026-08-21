import argparse
from pathlib import Path

import numpy as np
from cheartpy.abaqus.writer import import_region_mask
from cheartpy.gmsh import convert_3d_to_msh_via_api
from cheartpy.mesh import import_cheart_mesh
from cheartpy.mesh_tools.repair import fix_tetra_mesh

parser = argparse.ArgumentParser(description="Convert a 3D mesh to GMSH format.")
parser.add_argument("file", type=Path, help="Path to the 3D mesh file.")
parser.add_argument(
    "--mask", "--region-mask", "-m", type=Path, help="Path to the region mask file."
)
parser.add_argument(
    "--optimize",
    "--opt",
    action="store_true",
    help="Optimize the mesh using GMSH's built-in optimization algorithms.",
)
parser.add_argument("--save", action="store_true", help="Save the converted mesh to a file.")


def import_to_gmsh[I: np.integer](
    file: Path, mask: Path | None = None, *, save: bool = False, optimize: bool = False
) -> None:
    mesh = import_cheart_mesh(file).unwrap()
    mesh = fix_tetra_mesh(mesh).unwrap()
    regions = import_region_mask(mask) if mask else None
    filename = file.with_suffix(".inp") if save else None
    convert_3d_to_msh_via_api(mesh, regions=regions, filename=filename, optimize=optimize)


def main(cmdline: list[str] | None = None) -> None:
    args = parser.parse_args(cmdline)
    import_to_gmsh(args.file, args.mask, save=args.save, optimize=args.optimize)


if __name__ == "__main__":
    main()
