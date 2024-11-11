__all__ = ["get_space_data"]
import numpy as np
from ..cheart_mesh.io import *
from ..var_types import *

# from .interfaces import *

# from .variable_naming import *

# from .third_party import compress_vtu
# from concurrent import futures
# from ..var_types import i32, f64, Arr
# from ..cheart_mesh.io import *
# from ..xmlwriter.xmlclasses import XMLElement, XMLWriters

# # from ..cheart2vtu_core.print_headers import print_input_info
# # from ..cheart2vtu_core.main_parser import get_cmdline_args
# # from ..cheart2vtu_core.file_indexing import (
# #     IndexerList,
# #     get_file_name_indexer,
# # )
# # from ..cheart2vtu_core.data_types import (
# #     CheartMeshFormat,
# #     CheartVarFormat,
# #     CheartZipFormat,
# #     InputArguments,
# #     ProgramArgs,
# #     VariableCache,
# #     CheartTopology,
# # )
# from ..tools.progress_bar import ProgressBar
# from ..tools.parallel_exec import *
from .print_headers import *


def get_space_data(space_file: str | Mat[f64], disp_file: str | None) -> Mat[f64]:
    if isinstance(space_file, str):
        fx = CHRead_d(space_file)
    else:
        fx = space_file
    if disp_file is not None:
        fx = fx + CHRead_d_utf(disp_file)
    # VTU files are defined in 3D space, so we have to append a zero column for 2D data
    if fx.shape[1] == 1:
        raise ValueError(">>>ERROR: Cannot convert data that lives on 1D domains.")
    elif fx.shape[1] == 2:
        z = np.zeros((fx.shape[0], 3), dtype=float)
        z[:, :2] = fx
        return z
    return fx


def find_space_filenames(
    inp: ProgramArgs, time: int | str, cache: VariableCache
) -> tuple[Arr[tuple[int, int], f64] | str, None | str]:
    if isinstance(inp.space, CheartMeshFormat):
        fx = cache.nodes
    else:
        fx = inp.space.get_name(time)
        if os.path.isfile(fx):
            cache.space = fx
        else:
            fx = cache.space
    # if deformed space, then add displacement
    if inp.disp is None:
        return fx, None
    fd = inp.disp.get_name(time)
    if os.path.isfile(fd):
        cache.disp = fd
    else:
        fd = cache.disp
    return fx, fd


# def import_mesh_data(args: InputArguments, binary: bool = False):
#     fx = get_space_data(args.space, args.disp)
#     variables: dict[str, Arr[tuple[int, int], f64]] = dict()
#     for s, v in args.var.items():
#         # if binary:
#         #     fv = CHRead_d_bin(v)
#         # else:
#         #     fv = CHRead_d_utf(v)
#         fv = CHRead_d(v)
#         if fv.ndim == 1:
#             fv = fv[:, np.newaxis]
#         if fx.shape[0] != fv.shape[0]:
#             raise AssertionError(
#                 f">>>ERROR: Number of values for {
#                                  s} does not match Space."
#             )
#         # append zero column if need be
#         if fv.shape[1] == 2:
#             z = np.zeros((fv.shape[0], 3), dtype=float)
#             z[:, :2] = fv
#             variables[s] = z
#         else:
#             variables[s] = fv
#     return fx, variables
