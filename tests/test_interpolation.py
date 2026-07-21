"""
tests/test_interpolation.py — tests for numerix.core.interpolation (§6.4, §11).

Known-answer check per §11: "Interpolating a known polynomial exactly
(interpolation of a degree-2 function via quadratic interpolation
should be exact)". We use f(x) = 2x^2 - 3x + 1 throughout -- every
method here should reconstruct it exactly (up to floating point),
since none of them are asked to extrapolate degree beyond what the
data actually contains.
"""

import pytest

from numerix.core.interpolation import (
    cubic,
    lagrange,
    linear,
    newton_divided_diff,
    newton_gregory_backward,
    newton_gregory_forward,
    quadratic,
)


def f(x: float) -> float:
    return 2 * x**2 - 3 * x + 1


TOL = 1e-6

# Equally spaced sample points of f, x = 0, 1, 2, 3
XS4 = [0.0, 1.0, 2.0, 3.0]
YS4 = [f(x) for x in XS4]  # [1.0, 0.0, 3.0, 10.0]

# 3-point subset for quadratic (still exact, since f is degree 2)
XS3 = [0.0, 1.0, 2.0]
YS3 = [f(x) for x in XS3]  # [1.0, 0.0, 3.0]

# A genuinely linear function, for the linear-interpolation test
# (quadratic data would NOT be exact under plain linear interpolation).
def g(x: float) -> float:
    return 3 * x + 2


XS2 = [0.0, 5.0]
YS2 = [g(x) for x in XS2]  # [2.0, 17.0]


# ----------------------------------------------------------------------
# Known-answer / exactness tests
# ----------------------------------------------------------------------

def test_linear_exact_on_linear_function():
    x_target = 2.0
    result = linear(XS2, YS2, x_target)
    assert result.converged
    assert abs(result.solution - g(x_target)) < TOL
    assert result.method_name == "Linear Interpolation"
    assert result.category == "Interpolation"


def test_quadratic_exact_on_degree_2_polynomial():
    x_target = 1.5
    result = quadratic(XS3, YS3, x_target)
    assert result.converged
    assert abs(result.solution - f(x_target)) < TOL


def test_cubic_exact_on_degree_2_polynomial():
    # A cubic fit through 4 points sampled from a degree-2 function is
    # still exact -- the cubic (x^3) coefficient comes out as zero.
    x_target = 2.5
    result = cubic(XS4, YS4, x_target)
    assert result.converged
    assert abs(result.solution - f(x_target)) < TOL


def test_lagrange_exact_on_degree_2_polynomial():
    x_target = 1.5
    result = lagrange(XS3, YS3, x_target)
    assert result.converged
    assert abs(result.solution - f(x_target)) < TOL
    # Lagrange through the same 3 points should agree with `quadratic` exactly.
    assert abs(result.solution - quadratic(XS3, YS3, x_target).solution) < TOL


def test_newton_divided_diff_exact_on_degree_2_polynomial():
    x_target = 2.5
    result = newton_divided_diff(XS4, YS4, x_target)
    assert result.converged
    assert abs(result.solution - f(x_target)) < TOL


def test_newton_gregory_forward_exact_near_start():
    x_target = 0.5  # near the start of the (equally spaced) table
    result = newton_gregory_forward(XS4, YS4, x_target)
    assert result.converged
    assert abs(result.solution - f(x_target)) < TOL


def test_newton_gregory_backward_exact_near_end():
    x_target = 2.5  # near the end of the (equally spaced) table
    result = newton_gregory_backward(XS4, YS4, x_target)
    assert result.converged
    assert abs(result.solution - f(x_target)) < TOL


def test_all_methods_agree_on_same_data():
    # Lagrange, Newton divided difference, and both Newton-Gregory
    # forms are all different routes to the *same* interpolating
    # polynomial through XS4/YS4, so they must agree at any x_target.
    x_target = 1.75
    values = [
        lagrange(XS4, YS4, x_target).solution,
        newton_divided_diff(XS4, YS4, x_target).solution,
        newton_gregory_forward(XS4, YS4, x_target).solution,
        newton_gregory_backward(XS4, YS4, x_target).solution,
    ]
    for v in values:
        assert abs(v - f(x_target)) < TOL


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_linear_wrong_point_count_raises():
    with pytest.raises(ValueError):
        linear(XS3, YS3, 1.0)  # 3 points given, linear requires exactly 2


def test_quadratic_wrong_point_count_raises():
    with pytest.raises(ValueError):
        quadratic(XS2, YS2, 1.0)  # 2 points given, quadratic requires exactly 3


def test_cubic_wrong_point_count_raises():
    with pytest.raises(ValueError):
        cubic(XS3, YS3, 1.0)  # 3 points given, cubic requires exactly 4


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        lagrange([0.0, 1.0, 2.0], [1.0, 2.0], 1.0)  # xs/ys length mismatch


def test_duplicate_x_raises():
    with pytest.raises(ValueError):
        lagrange([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], 0.5)


def test_newton_gregory_forward_unequal_spacing_raises():
    with pytest.raises(ValueError):
        newton_gregory_forward([0.0, 1.0, 3.0], [1.0, 0.0, 10.0], 0.5)


def test_newton_gregory_backward_unequal_spacing_raises():
    with pytest.raises(ValueError):
        newton_gregory_backward([0.0, 1.0, 3.0], [1.0, 0.0, 10.0], 2.5)


def test_lagrange_too_few_points_raises():
    with pytest.raises(ValueError):
        lagrange([1.0], [2.0], 1.5)