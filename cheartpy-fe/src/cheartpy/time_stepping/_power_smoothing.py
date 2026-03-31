from typing import TYPE_CHECKING, TypedDict

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1, DType, ToFloat, ToInt

_TOL: float = 1e-2


class OptionalKwargs(TypedDict, total=False): ...


class StepSizes(TypedDict, total=True):
    left: ToFloat | None
    desired: ToFloat
    right: ToFloat | None


def _compute_optimal_step_factor(left: ToFloat, right: ToFloat, nt: ToInt) -> float:
    ratio = right / left
    factor = np.log(ratio) / nt
    return float(np.exp(factor))


def _compute_total_expected_ramp_time(left: ToFloat, factor: ToFloat, nt: ToInt) -> float:
    return float(left * factor * (factor**nt - 1) / (factor - 1))


def _define_ramp_steps[F: np.floating](
    left: ToFloat, factor: ToFloat, nt: ToInt, *, dtype: DType[F] = np.float64
) -> A1[F]:
    return (left * np.power(factor, np.arange(nt, dtype=dtype))).astype(dtype)


def _power_smooth_left[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    nt: ToInt,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    left = step_sizes["left"]
    desired = step_sizes["desired"]
    right = step_sizes["right"]
    if left is None:
        return _expand_dt_power(duration, step_sizes, nt, dtype=dtype)
    factor = _compute_optimal_step_factor(left, desired, nt)
    time_used = _compute_total_expected_ramp_time(left, factor, nt)
    ramp = _define_ramp_steps(left, factor, nt, dtype=dtype)
    match _expand_dt_power(
        duration - time_used,
        {"left": None, "desired": desired, "right": right},
        nt,
        dtype=dtype,
    ):
        case Ok(ramp_steps): ...  # fmt: skip
        case Err(err):
            return Err(err)
    return Ok([ramp, *ramp_steps])


def _power_smooth_right[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    nt: ToInt,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    left = step_sizes["left"]
    desired = step_sizes["desired"]
    right = step_sizes["right"]
    if right is None:
        return _expand_dt_power(duration, step_sizes, nt, dtype=dtype)
    factor = _compute_optimal_step_factor(desired, right, nt)
    time_used = _compute_total_expected_ramp_time(desired, factor, nt)
    ramp = _define_ramp_steps(desired * factor**nt, 1 / factor, nt, dtype=dtype)[::-1]
    match _expand_dt_power(
        duration - time_used,
        {"left": left, "desired": desired, "right": None},
        nt,
        dtype=dtype,
    ):
        case Ok(ramp_steps): ...  # fmt: skip
        case Err(err):
            return Err(err)
    return Ok([*ramp_steps, ramp])


def _expand_dt_power[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    nt: ToInt,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    if (step_sizes["left"]) and abs(step_sizes["desired"] - step_sizes["left"]) > _TOL:
        return _power_smooth_left(duration, step_sizes, nt, dtype=dtype).next()
    if (step_sizes["right"]) and abs(step_sizes["desired"] - step_sizes["right"]) > _TOL:
        return _power_smooth_right(duration, step_sizes, nt, dtype=dtype).next()
    better_nt = int(np.ceil(duration / step_sizes["desired"]))
    better_dt = duration / better_nt
    return Ok([np.full(better_nt, better_dt, dtype=dtype)])


def expand_timesteps_power[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    nt: ToInt,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    return _expand_dt_power(duration, step_sizes, nt, dtype=dtype).next()
