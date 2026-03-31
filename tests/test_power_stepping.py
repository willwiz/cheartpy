# pyright: reportUnknownMemberType=false
import math

import numpy as np
import pytest
from cheartpy.time_stepping import expand_timesteps_power


def test_left_expand() -> None:
    result = expand_timesteps_power(
        duration=10.0, step_sizes={"desired": 0.1, "left": 0.01, "right": 0.1000001}, nt=20
    ).unwrap()

    factor = np.exp(np.log(0.1 / 0.01) / 20)
    time = float(0.01 * (factor**20 - 1) / (factor - 1))
    expected_left = 0.01 * np.power(factor, np.arange(20))
    expected_right = np.array([])
    nt = math.ceil((10.0 - time) / 0.1)
    dt = (10.0 - time) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_right_contract() -> None:
    result = expand_timesteps_power(
        duration=10.0, step_sizes={"desired": 0.1, "left": None, "right": 0.01}, nt=20
    ).unwrap()

    factor = np.exp(np.log(0.1 / 0.01) / 20)
    time = float(0.01 * (factor**20 - 1) / (factor - 1))
    expected_left = np.array([])
    expected_right = 0.01 * np.power(factor, np.arange(20))[::-1]
    print(factor)
    print(expected_right)
    nt = math.ceil((10.0 - time) / 0.1)
    dt = (10.0 - time) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_both() -> None:
    result = expand_timesteps_power(
        duration=10.0, step_sizes={"desired": 0.1, "left": 0.01, "right": 1.0}, nt=10
    ).unwrap()

    left_factor = np.exp(np.log(0.1 / 0.01) / 10)
    left_time = float(0.01 * (left_factor**10 - 1) / (left_factor - 1))
    expected_left = 0.01 * np.power(left_factor, np.arange(10))
    right_factor = np.exp(np.log(0.1 / 1.0) / 10)
    right_time = float(1.0 * (right_factor**10 - 1) / (right_factor - 1))
    expected_right = 1.0 * np.power(right_factor, np.arange(10))[::-1]
    print(left_factor)
    print(expected_right)
    nt = int(np.ceil(((10.0 - expected_left.sum()) - expected_right.sum()) / 0.1))
    dt = ((10.0 - left_time) - right_time) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)
