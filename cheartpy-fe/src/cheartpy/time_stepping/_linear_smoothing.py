import math
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1, DType, ToFloat

_TOL: float = 1e-8


class OptionalKwargs(TypedDict, total=False): ...


class StepSizes(NamedTuple):
    left: ToFloat
    desired: ToFloat
    right: ToFloat


def define_ramp_steps[F: np.floating](
    left: ToFloat, right: ToFloat, ddt: ToFloat, *, dtype: DType[F] = np.float64
) -> A1[F]:
    return np.arange(left, right, math.copysign(ddt, right - left), dtype=dtype)


def compute_total_expected_ramp_time[F: np.floating](
    left: ToFloat, right: ToFloat, ddt: ToFloat
) -> float:
    nt = int(abs(right - left) / ddt)
    return float(nt * (left + right) / 2.0)


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
    if compute_total_expected_ramp_time(step_sizes.left, step_sizes.right, ddt) > duration:
        return Err(ValueError("Duration is too short to ramp up!"))
    left_ramp = define_ramp_steps(step_sizes.left, step_sizes.desired, ddt, dtype=dtype)
    remaining_duration = duration - left_ramp.sum()
    right_ramp = define_ramp_steps(step_sizes.desired, step_sizes.right, ddt, dtype=dtype)
    remaining_duration = remaining_duration - right_ramp.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Duration left between ramp is negative!"))
    plateau = np.full(
        int(remaining_duration / step_sizes.desired), float(step_sizes.desired), dtype=dtype
    )
    remaining_duration = remaining_duration - plateau.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Leftover time is negative!"))
    left_over = np.full(
        int(remaining_duration / step_sizes.left), float(step_sizes.left), dtype=dtype
    )
    return Ok([left_over, left_ramp, plateau, right_ramp])


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
    if compute_total_expected_ramp_time(step_sizes.left, step_sizes.right, ddt) < duration:
        return Err(ValueError("Duration is too short to ramp down!"))
    left_ramp = define_ramp_steps(step_sizes.left, step_sizes.desired, ddt, dtype=dtype)
    remaining_duration = duration - left_ramp.sum()
    right_ramp = define_ramp_steps(step_sizes.desired, step_sizes.right, ddt, dtype=dtype)
    remaining_duration = remaining_duration - right_ramp.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Duration left between ramp is negative!"))
    plateau = np.full(
        int(remaining_duration / step_sizes.desired), float(step_sizes.desired), dtype=dtype
    )
    remaining_duration = remaining_duration - plateau.sum()
    if remaining_duration < 0.0:
        return Err(ValueError("Leftover time is negative!"))
    left_over = np.full(
        int(remaining_duration / step_sizes.right), float(step_sizes.right), dtype=dtype
    )
    return Ok([left_ramp, plateau, right_ramp, left_over])


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
    raise NotImplementedError


def _trapazoid_down[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]: ...


def _expand_dt_linearly[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    if step_sizes.left < step_sizes.desired < step_sizes.right:
        return _double_up_ramp(duration, step_sizes, ddt, dtype=dtype).next()
    if step_sizes.left > step_sizes.desired > step_sizes.right:
        return _double_down_ramp(duration, step_sizes, ddt, dtype=dtype).next()
    if (step_sizes.desired < step_sizes.left) and (step_sizes.desired < step_sizes.right):
        return _trapazoid_down(duration, step_sizes, ddt, dtype=dtype).next()
    if (step_sizes.desired > step_sizes.left) and (step_sizes.desired > step_sizes.right):
        return _trapazoid_up(duration, step_sizes, ddt, dtype=dtype).next()
    msg = "Unreachable!"
    return Err(AssertionError(msg))


def expand_timesteps_linearly[F: np.floating](
    duration: ToFloat,
    step_sizes: StepSizes,
    ddt: ToFloat,
    *,
    dtype: DType[F] = np.float64,
) -> Result[Sequence[A1[F]]]:
    return _expand_dt_linearly(duration, step_sizes, ddt, dtype=dtype).next()
