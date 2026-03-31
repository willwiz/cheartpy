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
    return float(left * (factor**nt - 1) / (factor - 1))


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
    if time_used > duration:
        msg = f"Cannot fit ramp = {time_used} within duration = {duration} in {nt} steps"
        return Err(ValueError(msg))
    ramp = _define_ramp_steps(left, factor, nt, dtype=dtype)
    match _expand_dt_power(
        duration - ramp.sum(),
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
    factor = _compute_optimal_step_factor(right, desired, nt)
    time_used = _compute_total_expected_ramp_time(right, factor, nt)
    if time_used > duration:
        msg = f"Cannot fit ramp = {time_used} within duration = {duration} in {nt} steps"
        return Err(ValueError(msg))
    ramp = _define_ramp_steps(right, factor, nt, dtype=dtype)[::-1]
    match _expand_dt_power(
        duration - ramp.sum(),
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
    if (step_sizes["left"]) and _is_different(step_sizes["desired"], step_sizes["left"]):
        print("A")
        return _power_smooth_left(duration, step_sizes, nt, dtype=dtype).next()
    if (step_sizes["right"]) and _is_different(step_sizes["desired"], step_sizes["right"]):
        print("B")
        return _power_smooth_right(duration, step_sizes, nt, dtype=dtype).next()
    print("C")
    better_nt = int(np.ceil(duration / step_sizes["desired"]))
    better_dt = duration / better_nt
    return Ok([np.full(better_nt, better_dt, dtype=dtype)])


def _is_different(a: ToFloat, b: ToFloat) -> bool:
    return bool((max(a, b) / min(a, b)) > 1 + _TOL)


def expand_timesteps_power[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    nt: ToInt,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    return _expand_dt_power(duration, step_sizes, nt, dtype=dtype).next()
