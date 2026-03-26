import math
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1, DType, ToFloat

_TOL: float = 1e-8


class OptionalKwargs(TypedDict, total=False): ...


class StepSizes(TypedDict, total=True):
    left: ToFloat
    desired: ToFloat
    right: ToFloat


def define_ramp_steps[F: np.floating](
    left: ToFloat, right: ToFloat, ddt: ToFloat, *, dtype: DType[F] = np.float64
) -> A1[F]:
    return np.arange(left, right, math.copysign(ddt, right - left), dtype=dtype)


def compute_total_expected_ramp_time[F: np.floating](
    left: ToFloat, right: ToFloat, ddt: ToFloat, *, dtype: DType[F] = np.float64
) -> float:
    return np.arange(left, right, math.copysign(ddt, right - left), dtype=dtype).sum()


def _double_up_ramp[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    r"""Ramp up from left to desired, then ramp up from desired to right.

                      /
                    /
          ________/
        /
      /
    /
    """
    if compute_total_expected_ramp_time(step_sizes["left"], step_sizes["right"], ddt) > duration:
        return Err(ValueError("Duration is too short to ramp up!"))
    left_ramp = define_ramp_steps(step_sizes["left"], step_sizes["desired"], ddt, dtype=dtype)
    right_ramp = define_ramp_steps(step_sizes["desired"], step_sizes["right"], ddt, dtype=dtype)
    remaining_duration = duration - left_ramp.sum() - right_ramp.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Duration left between ramp is negative!"))
    nt = math.ceil(remaining_duration / step_sizes["desired"])
    best_dt = remaining_duration / nt
    plateau = np.full(nt, best_dt, dtype=dtype)
    return Ok([left_ramp, plateau, right_ramp])


def _double_down_ramp[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    r"""Ramp down from left to desired, then ramp down from desired to right.

    \
     \____
          \
           \
    """
    if compute_total_expected_ramp_time(step_sizes["left"], step_sizes["right"], ddt) > duration:
        return Err(ValueError("Duration is too short to ramp down!"))
    left_ramp = define_ramp_steps(step_sizes["left"], step_sizes["desired"], ddt, dtype=dtype)
    right_ramp = define_ramp_steps(step_sizes["desired"], step_sizes["right"], ddt, dtype=dtype)
    remaining_duration = duration - left_ramp.sum() - right_ramp.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Duration left between ramp is negative!"))
    nt = math.ceil(remaining_duration / step_sizes["desired"])
    best_dt = remaining_duration / nt
    plateau = np.full(nt, best_dt, dtype=dtype)
    return Ok([left_ramp, plateau, right_ramp])


def _trapazoid_up[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    r"""Ramp up from left, plateau, then ramp down to the right.

      ________
    /         \
               \
                \
    """
    left_duration = compute_total_expected_ramp_time(step_sizes["left"], step_sizes["desired"], ddt)
    right_duration = compute_total_expected_ramp_time(
        step_sizes["desired"], step_sizes["right"], ddt
    )
    if left_duration + right_duration > duration:
        return Err(ValueError("Duration is too short to ramp up and down!"))
    left_ramp = define_ramp_steps(step_sizes["left"], step_sizes["desired"], ddt, dtype=dtype)
    right_ramp = define_ramp_steps(step_sizes["desired"], step_sizes["right"], ddt, dtype=dtype)
    remaining_duration = duration - left_ramp.sum() - right_ramp.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Duration left between ramp is negative!"))
    nt = math.ceil(remaining_duration / step_sizes["desired"])
    best_dt = remaining_duration / nt
    plateau = np.full(nt, best_dt, dtype=dtype)
    return Ok([left_ramp, plateau, right_ramp])


def _trapazoid_down[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    r"""Ramp down from left, plateau, then ramp up to the right.

    \         /
     \_______/

    """
    left_duration = compute_total_expected_ramp_time(step_sizes["left"], step_sizes["desired"], ddt)
    right_duration = compute_total_expected_ramp_time(
        step_sizes["desired"], step_sizes["right"], ddt
    )
    if left_duration + right_duration > duration:
        return Err(ValueError("Duration is too short to ramp down and up!"))
    left_ramp = define_ramp_steps(step_sizes["left"], step_sizes["desired"], ddt, dtype=dtype)
    right_ramp = define_ramp_steps(step_sizes["desired"], step_sizes["right"], ddt, dtype=dtype)
    remaining_duration = duration - left_ramp.sum() - right_ramp.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Duration left between ramp is negative!"))
    nt = math.ceil(remaining_duration / step_sizes["desired"])
    best_dt = remaining_duration / nt
    plateau = np.full(nt, best_dt, dtype=dtype)
    return Ok([left_ramp, plateau, right_ramp])


def _expand_dt_linearly[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    if step_sizes["left"] <= step_sizes["desired"] <= step_sizes["right"]:
        return _double_up_ramp(duration, step_sizes, ddt, dtype=dtype).next()
    if step_sizes["left"] >= step_sizes["desired"] > step_sizes["right"]:
        return _double_down_ramp(duration, step_sizes, ddt, dtype=dtype).next()
    if (step_sizes["desired"] <= step_sizes["left"]) and (
        step_sizes["desired"] <= step_sizes["right"]
    ):
        return _trapazoid_down(duration, step_sizes, ddt, dtype=dtype).next()
    if (step_sizes["desired"] >= step_sizes["left"]) and (
        step_sizes["desired"] >= step_sizes["right"]
    ):
        return _trapazoid_up(duration, step_sizes, ddt, dtype=dtype).next()
    msg = "Unreachable!"
    return Err(AssertionError(msg))


def expand_timesteps_linearly[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    ddt = min([step_sizes["desired"], step_sizes["left"], step_sizes["right"]])
    return _expand_dt_linearly(duration, step_sizes, ddt, dtype=dtype).next()
