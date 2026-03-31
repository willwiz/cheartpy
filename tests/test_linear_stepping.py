# pyright: reportUnknownMemberType=false
import math

import numpy as np
import pytest
from cheartpy.time_stepping import expand_timesteps_linearly


def test_left_expand() -> None:
    result = expand_timesteps_linearly(
        duration=10.0, step_sizes={"desired": 0.1, "left": 0.01, "right": 0.1}
    ).unwrap()
    expected_left = np.arange(0.01, 0.1, 0.01)
    expected_right = np.array([])
    duration = 10.0 - expected_left.sum() - expected_right.sum()
    nt = math.ceil((duration) / 0.1)
    dt = (duration) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    print(result)
    print(expected)
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert result.sum() == pytest.approx(10.0)
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_right_contract() -> None:
    result = expand_timesteps_linearly(
        duration=10.0, step_sizes={"desired": 0.1, "left": 0.1, "right": 0.01}
    ).unwrap()
    expected_left = np.array([])
    expected_right = np.arange(0.01, 0.1, 0.01)[::-1]
    duration = 10.0 - expected_left.sum() - expected_right.sum()
    nt = math.ceil((duration) / 0.1)
    dt = (duration) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    print(result)
    print(expected)
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert result.sum() == pytest.approx(10.0)
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_left() -> None:
    result = expand_timesteps_linearly(
        duration=10.0, step_sizes={"desired": 0.1, "left": 0.01, "right": 0.2}
    ).unwrap()
    expected_left = np.arange(0.01, 0.1, 0.01, dtype=np.float64)
    expected_right = np.arange(0.2, 0.1, -0.01, dtype=np.float64)[::-1]
    duration = 10.0 - expected_left.sum() - expected_right.sum()
    nt = math.ceil((duration) / 0.1)
    dt = (duration) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    print(result)
    print(expected)
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert result.sum() == pytest.approx(10.0)
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_right() -> None:
    result = expand_timesteps_linearly(
        duration=10.0, step_sizes={"desired": 0.1, "left": 0.2, "right": 0.01}
    ).unwrap()
    expected_left = np.arange(0.2, 0.1, -0.01, dtype=np.float64)
    expected_right = np.arange(0.01, 0.1, 0.01, dtype=np.float64)[::-1]
    duration = 10.0 - expected_left.sum() - expected_right.sum()
    nt = math.ceil((duration) / 0.1)
    dt = (duration) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    print(result)
    print(expected)
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert result.sum() == pytest.approx(10.0)
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_up() -> None:
    result = expand_timesteps_linearly(
        duration=10.0, step_sizes={"desired": 0.1, "left": 0.01, "right": 0.01}
    ).unwrap()
    expected_left = np.arange(0.01, 0.1, 0.01)
    expected_right = np.arange(0.01, 0.1, 0.01)[::-1]
    duration = 10.0 - expected_left.sum() - expected_right.sum()
    nt = math.ceil((duration) / 0.1)
    dt = (duration) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    print(result)
    print(expected)
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert result.sum() == pytest.approx(10.0)
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_down() -> None:
    result = expand_timesteps_linearly(
        duration=10.0, step_sizes={"desired": 0.01, "left": 0.1, "right": 0.1}
    ).unwrap()
    expected_left = np.arange(0.1, 0.01, -0.01)
    expected_right = np.arange(0.1, 0.01, -0.01)[::-1]
    duration = 10.0 - expected_left.sum() - expected_right.sum()
    nt = math.ceil((duration) / 0.01)
    dt = (duration) / nt
    expected = [expected_left, np.full(nt, dt), expected_right]
    print(result)
    print(expected)
    result = np.concatenate(result)
    print(result.sum())
    expected = np.concatenate(expected)
    print(expected.sum())
    assert result.sum() == pytest.approx(10.0)
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)
