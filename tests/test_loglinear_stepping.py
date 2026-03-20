# pyright: reportUnknownMemberType=false
import numpy as np
import pytest
from cheartpy.time_stepping import expand_time_as_log_linear


def test_left_expand() -> None:
    result = expand_time_as_log_linear(
        duration=10.0,
        desired=0.1,
        left=0.01,
        right=0.1,
        repeat_contract=1,
    ).unwrap()
    expected = [np.full(5, 0.01), np.arange(0.01, 0.1, 0.01), np.full(int((10.0 - 0.5) / 0.1), 0.1)]
    print(result)
    print(expected)
    result = np.concatenate(result)
    expected = np.concatenate(expected)
    assert result.sum() == pytest.approx(10.0)
    assert expected.sum() == pytest.approx(10.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_right_expand() -> None:
    result = expand_time_as_log_linear(
        duration=10.0,
        desired=0.1,
        left=0.1,
        right=0.01,
        repeat_contract=1,
    ).unwrap()
    expected = [
        np.full(int((10.0 - 0.5) / 0.1), 0.1),
        np.flip(np.arange(0.01, 0.1, 0.01)),
        np.full(5, 0.01),
    ]
    print(result)
    print(expected)
    result = np.concatenate(result)
    expected = np.concatenate(expected)
    assert result.sum() == pytest.approx(10.0, 1e-16)
    assert expected.sum() == pytest.approx(10.0, 1e-16)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_left_contract() -> None:
    result = expand_time_as_log_linear(
        duration=1.0,
        desired=0.01,
        left=0.1,
        right=0.01,
        repeat_contract=1,
    ).unwrap()
    expected = [np.flip(np.arange(0.01, 0.1, 0.01)), np.full(55, 0.01)]
    print(result)
    print(expected)
    result = np.concatenate(result)
    expected = np.concatenate(expected)
    assert result.sum() == pytest.approx(1.0)
    assert expected.sum() == pytest.approx(1.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)


def test_right_contract() -> None:
    result = expand_time_as_log_linear(
        duration=1.0,
        desired=0.01,
        left=0.01,
        right=0.1,
        repeat_contract=1,
    ).unwrap()
    expected = [np.full(55, 0.01), np.arange(0.01, 0.1, 0.01)]
    print(result)
    print(expected)
    result = np.concatenate(result)
    expected = np.concatenate(expected)
    assert result.sum() == pytest.approx(1.0)
    assert expected.sum() == pytest.approx(1.0)
    assert result.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(result, expected, atol=1e-16)
