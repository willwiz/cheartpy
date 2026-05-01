import numpy as np
from cheartpy.cl.mesh import CLDef, CLPartition
from pytools.arrays import A1, A2

def interp_cl_var_to_volume[F: np.floating, I: np.integer](
    a_z: A1[F], part: CLDef[F] | CLPartition[F], *v: A2[F]
) -> list[A2[F]]: ...
