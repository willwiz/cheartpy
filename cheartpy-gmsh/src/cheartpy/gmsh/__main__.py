import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io import chwrite_d_utf
from cheartpy.mesh import import_cheart_mesh
from cheartpy.mesh_tools.repair import fix_tetra_mesh

import gmsh

from .reader import gmsh_finalize, import_region_mask, read_cheartmesh_into_gmsh_api
from .writter import build_cheart_mesh_from_gmsh

if TYPE_CHECKING:
    from cheartpy.gmsh.types import GmshMeshTags

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
parser.add_argument(
    "--save", "--export", "-s", type=Path, help="Save the converted mesh to a file."
)


def export_mesh(tags: GmshMeshTags, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    match filename.suffix:
        case ".msh" | ".inp":
            gmsh.write(str(filename))
        case ".xtb":
            mesh, mask = build_cheart_mesh_from_gmsh(tags.dim, tags.domains, tags.boundarys)
            mesh.save(filename.name)
            if mask:
                chwrite_d_utf(filename.parent / f"{filename.stem}_mask-0.D", mask)
        case _:
            msg = f"Unsupported file format: {filename.suffix}"
            raise ValueError(msg)


def import_to_gmsh[I: np.integer](
    file: Path, domains: Path | None = None, *, save: Path | None = None, optimize: bool = False
) -> None:
    mesh = import_cheart_mesh(file).unwrap()
    mesh = fix_tetra_mesh(mesh).unwrap()
    regions = import_region_mask(domains) if domains else None
    filename = file.with_suffix(".inp") if save else None
    gmsh.initialize()
    tags = read_cheartmesh_into_gmsh_api(mesh, regions=regions, optimize=optimize)
    if save:
        export_mesh(tags, save)
    else:
        gmsh.fltk.run()
    gmsh_finalize(filename=filename)
    gmsh.finalize()


def main(cmdline: list[str] | None = None) -> None:
    args = parser.parse_args(cmdline)
    import_to_gmsh(args.file, args.mask, save=args.save, optimize=args.optimize)


if __name__ == "__main__":
    main()
