import math
from typing import TYPE_CHECKING, TypedDict, Unpack

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1, DType, ToFloat

_TOL = 1e-8


class OptionalKwargs[F: np.floating](TypedDict, total=False):
    repeat_expand: int
    repeat_contract: int


def dt_accelerate_base10[F: np.floating](
    dt: ToFloat, *, dtype: DType[F] = np.float64, repeat: int = 1
) -> A1[F]:
    return np.concatenate([np.full(repeat, dt * i) for i in range(1, 10)], dtype=dtype)


def dt_decelerate_base10[F: np.floating](
    dt: ToFloat, *, dtype: DType[F] = np.float64, repeat: int = 1
) -> A1[F]:
    dt = 0.1 * dt
    return np.concatenate([np.full(repeat, dt * i) for i in reversed(range(1, 10))], dtype=dtype)


def _expand_dt[F: np.floating](
    duration: float,
    desired: float,
    left: float,
    right: float,
    *,
    dtype: DType[F] = np.float64,
    **kwargs: Unpack[OptionalKwargs[F]],
) -> Result[Sequence[A1[F]]]:
    """Try to smoothly expand array dt to match desired while preserving the boundary values."""
    if float(desired) - float(left) > _TOL:
        return _left_expand(duration, desired, left, right, dtype=dtype, **kwargs).next()
    if float(left) - float(desired) > _TOL:
        return _left_contract(duration, desired, left, right, dtype=dtype, **kwargs).next()
    if float(desired) - float(right) > _TOL:
        return _right_contract(duration, desired, left, right, dtype=dtype, **kwargs).next()
    if float(right) - float(desired) > _TOL:
        return _right_expand(duration, desired, left, right, dtype=dtype, **kwargs).next()
    return Ok([np.full(int(duration / desired), float(desired), dtype=dtype)])


def _left_contract[F: np.floating](
    duration: float,
    desired: float,
    left: float,
    right: float,
    *,
    dtype: DType[F] = np.float64,
    **kwargs: Unpack[OptionalKwargs[F]],
) -> Result[Sequence[A1[F]]]:
    repeat = kwargs.get("repeat_contract", 1)
    ramp_t = 4.5 * repeat * left
    # print(f"_left_contract with parameters:{duration=}, {desired=}, {left=}, {right=}, {ramp_t=}")
    if duration < ramp_t:
        msg = (
            "Duration is too short to accommodate the desired time step from left!\n"
            f"_left_contract with parameters:{duration=}, {desired=}, {left=}, {right=}, {ramp_t=}"
        )
        return Err(ValueError(msg))
    match _expand_dt(duration - ramp_t, desired, 0.1 * left, right, dtype=dtype, **kwargs):
        case Ok(result): ...  # fmt: skip
        case Err(err):
            return Err(err)
    left_over = duration - sum([t.sum() for t in result]) - ramp_t
    if left_over < -_TOL:
        msg = (
            f"Expanded time steps exceed duration! Got {duration - left_over}, expected {duration}."
        )
        return Err(AssertionError(msg))
    left_over = left_over + _TOL
    return Ok(
        [
            dt_decelerate_base10(left, dtype=dtype, repeat=repeat),
            np.full(int(left_over / desired), float(desired), dtype=dtype),
            *result,
        ]
    )


def _left_expand[F: np.floating](
    duration: float,
    desired: float,
    left: float,
    right: float,
    *,
    dtype: DType[F] = np.float64,
    **kwargs: Unpack[OptionalKwargs[F]],
) -> Result[Sequence[A1[F]]]:
    repeat = kwargs.get("repeat_expand", 1)
    ramp_t = 45.0 * repeat * left
    # print(f"_left_expand with parameters:{duration=}, {desired=}, {left=}, {right=}, {ramp_t=}")
    if duration < ramp_t:
        # print(
        #     f"_left_expand: not enough time\n"
        #     f"from left! {duration} < {ramp_t}\n"
        #     f"changing desired from {desired} to {left} for the rest of the duration"
        # )
        return _expand_dt(duration, left, left, right, dtype=dtype, **kwargs).next()
    match _expand_dt(duration - ramp_t, desired, 10 * left, right, dtype=dtype, **kwargs):
        case Ok(result): ...  # fmt: skip
        case Err(err):
            return Err(err)
    left_over = duration - sum([t.sum() for t in result]) - ramp_t
    if left_over < -_TOL:
        msg = (
            f"Expanded time steps exceed duration! Got {duration - left_over}, expected {duration}."
        )
        return Err(AssertionError(msg))
    left_over = left_over + _TOL
    return Ok(
        [
            np.full(int(left_over / left), float(left), dtype=dtype),
            dt_accelerate_base10(left, dtype=dtype, repeat=repeat),
            *result,
        ]
    )


def _right_contract[F: np.floating](
    duration: float,
    desired: float,
    left: float,
    right: float,
    *,
    dtype: DType[F] = np.float64,
    **kwargs: Unpack[OptionalKwargs[F]],
) -> Result[Sequence[A1[F]]]:
    repeat = kwargs.get("repeat_contract", 1)
    ramp_t = 45.0 * repeat * right
    if duration < ramp_t:
        return Ok([np.full(int(duration / right), float(right), dtype=dtype)])
    match _expand_dt(duration - ramp_t, desired, left, 10.0 * right, dtype=dtype, **kwargs):
        case Ok(result): ...  # fmt: skip
        case Err(err):
            return Err(err)
    left_over = duration - sum([t.sum() for t in result]) - ramp_t
    if left_over < -_TOL:
        msg = (
            f"Expanded time steps exceed duration! Got {duration - left_over}, expected {duration}."
        )
        return Err(AssertionError(msg))
    left_over = left_over + _TOL
    return Ok(
        [
            *result,
            np.flip(dt_accelerate_base10(right, dtype=dtype, repeat=repeat)),
            np.full(int(left_over / right), float(right), dtype=dtype),
        ]
    )


def _right_expand[F: np.floating](
    duration: float,
    desired: float,
    left: float,
    right: float,
    *,
    dtype: DType[F] = np.float64,
    **kwargs: Unpack[OptionalKwargs[F]],
) -> Result[Sequence[A1[F]]]:
    repeat = kwargs.get("repeat_contract", 1)
    ramp_t = 4.5 * repeat * right
    # print(f"_right_expand with parameters:{duration=}, {desired=}, {left=}, {right=}, {ramp_t=}")
    if duration < ramp_t:
        msg = (
            "Duration is too short to accommodate the desired time step from right!\n"
            f"  Right side requires dt={right},\n"
            f"  want steps to be {desired}\n"
            f"  but duration is only {duration}."
        )
        return _expand_dt(duration, desired, left, desired, dtype=dtype, **kwargs).next()
    match _expand_dt(duration - ramp_t, desired, left, 0.1 * right, dtype=dtype, **kwargs):
        case Ok(result): ...  # fmt: skip
        case Err(err):
            return Err(err)
    left_over = duration - sum([t.sum() for t in result]) - ramp_t
    if left_over < -_TOL:
        msg = (
            f"Expanded time steps exceed duration! Got {duration - left_over}, expected {duration}."
        )
        return Err(AssertionError(msg))
    left_over = left_over + _TOL
    return Ok(
        [
            *result,
            np.full(int(10.0 * left_over / right), float(0.1 * right), dtype=dtype),
            np.flip(dt_decelerate_base10(right, dtype=dtype, repeat=repeat)),
        ]
    )


def expand_time_as_log_linear[F: np.floating](
    duration: ToFloat,
    desired: ToFloat,
    left: ToFloat,
    right: ToFloat,
    *,
    dtype: DType[F] = np.float64,
    **kwargs: Unpack[OptionalKwargs[F]],
) -> Result[Sequence[A1[F]]]:
    """Expand time steps to match desired while preserving the boundary values.

    Only accepts dt in powers of 10.
    """
    if desired == left == right:
        return Ok([np.full(int(duration / desired), float(desired), dtype=dtype)])
    # print(f"Arguments are: duration={duration}, desired={desired}, left={left}, right={right}")
    left = float(10 ** math.floor(math.log10(left)))
    right = float(10 ** math.floor(math.log10(right)))
    desired = float(10 ** math.floor(math.log10(desired)))
    # print(f"Updated time steps are: desired={desired}, left={left}, right={right}")
    match _expand_dt(float(duration), desired, left, right, dtype=dtype, **kwargs):
        case Ok(dt): ...  # fmt: skip
        case Err(err):
            return Err(err)
    current_duration = sum([t.sum() for t in dt])
    extra_duration = float(duration) - current_duration
    if extra_duration < -_TOL:
        msg = f"Expanded time steps exceed duration! Got {current_duration}, expected {duration}."
        return Err(AssertionError(msg))
    if extra_duration > _TOL:
        if left < right:
            dt = [np.full(int(extra_duration / left), left, dtype=dtype), *dt]
        else:
            dt = [*dt, np.full(int(extra_duration / right), right, dtype=dtype)]
    return Ok(dt)
