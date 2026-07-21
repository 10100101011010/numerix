"""
tests/test_integration.py — tests for numerix.core.integration (§6.5, §11).

Known-answer check per §11: integral of x**2 from 0 to 1 = 1/3,
checking Simpson's rule lands closer to the true value than the
rectangle rule for the same n.
"""

import pytest

from numerix.core.integration import (
    midpoint_rule,
    rectangle_rule,
    simpson_1_3,
    simpson_3_8,
    trapezoidal_rule,
)

TRUE_VALUE = 1.0 / 3.0  # integral of x**2 from 0 to 1


# ----------------------------------------------------------------------
# Known-answer tests
# ----------------------------------------------------------------------

def test_rectangle_rule_converges_toward_true_value():
    result = rectangle_rule("x**2", 0.0, 1.0, n=1000, side="left")
    assert result.converged
    assert abs(result.solution - TRUE_VALUE) < 1e-2
    assert result.method_name == "Rectangle Rule"
    assert result.category == "Numerical Integration"


def test_midpoint_rule_converges_toward_true_value():
    result = midpoint_rule("x**2", 0.0, 1.0, n=100)
    assert result.converged
    assert abs(result.solution - TRUE_VALUE) < 1e-4


def test_trapezoidal_rule_converges_toward_true_value():
    result = trapezoidal_rule("x**2", 0.0, 1.0, n=100)
    assert result.converged
    assert abs(result.solution - TRUE_VALUE) < 1e-4


def test_simpson_1_3_exact_on_quadratic():
    # Simpson's 1/3 rule is exact for polynomials up to degree 3, so
    # x**2 should be reproduced to (near) machine precision at ANY
    # valid (even) n, including the smallest one, n=2.
    result = simpson_1_3("x**2", 0.0, 1.0, n=2)
    assert result.converged
    assert abs(result.solution - TRUE_VALUE) < 1e-10
    assert result.method_name == "Simpson's 1/3 Rule"


def test_simpson_3_8_exact_on_quadratic():
    # Same exactness property for Simpson's 3/8 rule, at its smallest
    # valid n, n=3.
    result = simpson_3_8("x**2", 0.0, 1.0, n=3)
    assert result.converged
    assert abs(result.solution - TRUE_VALUE) < 1e-10
    assert result.method_name == "Simpson's 3/8 Rule"


def test_simpson_closer_than_rectangle_for_same_n():
    # §11's explicit required comparison.
    n = 10
    rect = rectangle_rule("x**2", 0.0, 1.0, n=n, side="left")
    simpson = simpson_1_3("x**2", 0.0, 1.0, n=n)

    rect_error = abs(rect.solution - TRUE_VALUE)
    simpson_error = abs(simpson.solution - TRUE_VALUE)

    assert simpson_error < rect_error


def test_simpson_3_8_closer_than_rectangle_for_same_n():
    n = 9  # divisible by 3
    rect = rectangle_rule("x**2", 0.0, 1.0, n=n, side="left")
    simpson38 = simpson_3_8("x**2", 0.0, 1.0, n=n)

    assert abs(simpson38.solution - TRUE_VALUE) < abs(rect.solution - TRUE_VALUE)


# ----------------------------------------------------------------------
# Basic correctness sanity checks (functions of low enough degree that
# each rule's own theoretical exactness bound applies, independent of
# any truncation-error tolerance)
# ----------------------------------------------------------------------

def test_all_methods_exact_on_constant_function():
    # A constant function has zero error under every rule, at any n.
    for method, kwargs in [
        (rectangle_rule, {"side": "left"}),
        (rectangle_rule, {"side": "right"}),
        (midpoint_rule, {}),
        (trapezoidal_rule, {}),
    ]:
        result = method("3", 0.0, 2.0, n=4, **kwargs)
        assert abs(result.solution - 6.0) < 1e-9

    assert abs(simpson_1_3("3", 0.0, 2.0, n=4).solution - 6.0) < 1e-9
    assert abs(simpson_3_8("3", 0.0, 2.0, n=3).solution - 6.0) < 1e-9


def test_trapezoidal_and_midpoint_exact_on_linear_function():
    # Both trapezoidal and midpoint rules are exact for degree <= 1.
    trap = trapezoidal_rule("2*x + 1", 0.0, 4.0, n=5)
    mid = midpoint_rule("2*x + 1", 0.0, 4.0, n=5)
    true_value = 20.0  # integral of (2x+1) from 0 to 4 = [x^2 + x] = 16 + 4 = 20
    assert abs(trap.solution - true_value) < 1e-9
    assert abs(mid.solution - true_value) < 1e-9


def test_rectangle_left_vs_right_differ_on_monotonic_function():
    left = rectangle_rule("x**2", 0.0, 1.0, n=10, side="left")
    right = rectangle_rule("x**2", 0.0, 1.0, n=10, side="right")
    # x**2 is strictly increasing, so left sum underestimates and right overestimates.
    assert left.solution < TRUE_VALUE < right.solution


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_invalid_bounds_raises():
    with pytest.raises(ValueError):
        rectangle_rule("x**2", 1.0, 0.0, n=10)  # b <= a


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        midpoint_rule("x**2", 0.0, 1.0, n=0)


def test_rectangle_invalid_side_raises():
    with pytest.raises(ValueError):
        rectangle_rule("x**2", 0.0, 1.0, n=10, side="middle")


def test_simpson_1_3_odd_n_raises():
    with pytest.raises(ValueError):
        simpson_1_3("x**2", 0.0, 1.0, n=5)


def test_simpson_3_8_non_multiple_of_3_raises():
    with pytest.raises(ValueError):
        simpson_3_8("x**2", 0.0, 1.0, n=4)