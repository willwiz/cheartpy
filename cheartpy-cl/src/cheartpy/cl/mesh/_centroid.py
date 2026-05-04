from typing import TYPE_CHECKING, Required, Unpack, cast

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from cheartpy.mesh import CheartMesh
    from pytools.arrays import A1, A2, ToFloat


class CentroidAPIKwargs[F: np.floating](TypedDict, total=False):
    width: float
    a_z: Required[A1[F]]
    v_z: Required[A2[F]]


def _mesh_centroid[F: np.floating, I: np.integer](mesh: CheartMesh[F, I]) -> A1[F]:
    return mesh.space.v.mean(axis=0)


def _construct_basis_at_z[F: np.floating, I: np.integer](
    z: ToFloat, **kwargs: Unpack[CentroidAPIKwargs[F]]
) -> tuple[A1[np.bool_], A1[F]]:
    a_z = kwargs["a_z"]
    width = kwargs.get("width") or 0.05
    domain = (a_z > (z - width)) & (a_z < (z + width))
    basis = (a_z[domain] - z + width) / width
    return domain, np.minimum(basis, 2.0 - basis)


def _compute_centroid_at_z[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    z: ToFloat,
    centroid: A1[F],
    **kwargs: Unpack[CentroidAPIKwargs[F]],
) -> A1[F]:
    domain, basis = _construct_basis_at_z(z, **kwargs)
    return (mesh.space.v[domain] * basis[:, None]).sum(axis=0) / basis.sum() - centroid


def _compute_a_c_coordinate_at_z[F: np.floating](v_z: A1[F], v_r: A1[F], v_ref: A1[F]) -> F:
    v_0 = v_ref - (v_z @ v_ref) * v_z
    v_0 = v_0 / np.linalg.norm(v_0) if np.linalg.norm(v_0) > 0 else v_z
    v_p = v_r - (v_z @ v_r) * v_z
    v_p = v_p / np.linalg.norm(v_p) if np.linalg.norm(v_p) > 0 else v_z
    quat, *_ = Rotation.align_vectors(v_0, v_p)
    rot = quat.as_rotvec()
    q = np.linalg.norm(rot)
    axis = (rot / q).astype(v_z.dtype) if q > 0 else v_z
    q = q * np.sign(np.dot(axis, v_z))
    return q % (2 * np.pi)


def compute_a_c_coordinate[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], **kwargs: Unpack[CentroidAPIKwargs[F]]
) -> A1[F]:
    """Compute the a_c coordinate of the volume.

    a_c coordinate is defined as the angle between the coordinate of the centerline and the
    coordinate of the node relative to the centerline.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh.
    a_z : A1[F]
        The centerline a_z coordinate of the volume.
    v_z : A2[F]
        The local v_z axis.
    width : float, optional
        The width of the basis function, by default 0.05

    Returns
    -------
    A1[F]
        The a_c coordinate of the volume.

    """
    a_z = kwargs["a_z"]
    centroid = _mesh_centroid(mesh)
    ar_gen = (_compute_centroid_at_z(mesh, z, centroid, **kwargs) for z in a_z)
    v_ref = np.fromiter(ar_gen, dtype=np.dtype((a_z.dtype, 3)), count=len(a_z))
    v_ref = cast("A2[F]", v_ref)
    v_p = mesh.space.v - v_ref
    ac_gen = (
        _compute_a_c_coordinate_at_z(z, r, c)
        for z, r, c in zip(kwargs["v_z"], v_p, v_ref, strict=True)
    )
    return np.fromiter(ac_gen, dtype=a_z.dtype, count=len(a_z))
