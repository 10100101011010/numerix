"""
tests/test_nonlinear.py — tests for numerix.core.nonlinear (§6.2, §11).

Known-answer check per §11: the root of x**3 - x - 2 is ~1.5214.
Bisection, regula falsi, fixed point, Newton-Raphson, and secant
should all land there. Also covers the §11 edge cases explicitly
called out for this module: non-convergence (max_iter hit without
meeting tolerance) and invalid bracket for bisection/regula falsi --
plus the zero-derivative / zero-denominator error paths that §6.2
requires of Newton-Raphson and secant.
"""

import pytest

from numerix.core.nonlinear import (
    bisection,
    fixed_point,
    newton_raphson,
    regula_falsi,
    secant,
)

F_TEXT = "x**3 - x - 2"
TRUE_ROOT = 1.5213797068045675
ROOT_TOL = 1e-3


# ----------------------------------------------------------------------
# Known root of x**3 - x - 2 ~= 1.5214, all five methods
# ----------------------------------------------------------------------

def test_bisection_known_root():
    result = bisection(F_TEXT, a=1.0, b=2.0, tol=1e-8, max_iter=100)
    assert result.converged
    assert abs(result.solution - TRUE_ROOT) < ROOT_TOL
    assert result.method_name == "Bisection"
    assert result.category == "Nonlinear Equations"
    assert result.n_iterations > 0
    assert len(result.iterations) == result.n_iterations


def test_regula_falsi_known_root():
    result = regula_falsi(F_TEXT, a=1.0, b=2.0, tol=1e-8, max_iter=200)
    assert result.converged
    assert abs(result.solution - TRUE_ROOT) < ROOT_TOL
    assert result.method_name == "Regula Falsi"


def test_fixed_point_known_root():
    # x = (x + 2)**(1/3) is a rearrangement of x**3 - x - 2 = 0 that
    # converges (|g'(x)| < 1) near x0 = 1.5.
    result = fixed_point("(x + 2)**(1/3)", x0=1.5, tol=1e-8, max_iter=200)
    assert result.converged
    assert abs(result.solution - TRUE_ROOT) < ROOT_TOL
    assert result.method_name == "Fixed Point"
    assert result.warning is None  # |g'(1.5)| < 1 here, so no warning expected


def test_newton_raphson_known_root():
    result = newton_raphson(F_TEXT, x0=1.5, tol=1e-8, max_iter=100)
    assert result.converged
    assert abs(result.solution - TRUE_ROOT) < ROOT_TOL
    assert result.method_name == "Newton-Raphson"
    assert result.n_iterations < 10  # quadratic convergence, should be fast


def test_secant_known_root():
    result = secant(F_TEXT, x0=1.0, x1=2.0, tol=1e-8, max_iter=100)
    assert result.converged
    assert abs(result.solution - TRUE_ROOT) < ROOT_TOL
    assert result.method_name == "Secant"


# ----------------------------------------------------------------------
# §11 edge case: non-convergence (max_iter hit without meeting tol)
# ----------------------------------------------------------------------

def test_bisection_non_convergence_reports_gracefully():
    # An unreachably tight tolerance with a tiny max_iter must NOT
    # raise -- it should come back as an ordinary MethodResult with
    # converged=False, per §5.
    result = bisection(F_TEXT, a=1.0, b=2.0, tol=1e-15, max_iter=3)
    assert result.converged is False
    assert result.n_iterations == 3
    assert len(result.iterations) == 3
    assert result.solution is not None


def test_regula_falsi_non_convergence_reports_gracefully():
    result = regula_falsi(F_TEXT, a=1.0, b=2.0, tol=1e-15, max_iter=3)
    assert result.converged is False
    assert result.n_iterations == 3


def test_fixed_point_non_convergence_reports_gracefully():
    result = fixed_point("(x + 2)**(1/3)", x0=1.5, tol=1e-15, max_iter=3)
    assert result.converged is False
    assert result.n_iterations == 3


def test_newton_raphson_non_convergence_reports_gracefully():
    result = newton_raphson(F_TEXT, x0=1.5, tol=1e-15, max_iter=2)
    assert result.converged is False
    assert result.n_iterations == 2


def test_secant_non_convergence_reports_gracefully():
    result = secant(F_TEXT, x0=1.0, x1=2.0, tol=1e-15, max_iter=2)
    assert result.converged is False
    assert result.n_iterations == 2


# ----------------------------------------------------------------------
# §11 edge case: invalid bracket for bisection/regula falsi
# ----------------------------------------------------------------------

def test_bisection_invalid_bracket_raises():
    # f(-5) and f(-3) are both negative for x**3 - x - 2 -- no sign change.
    with pytest.raises(ValueError):
        bisection(F_TEXT, a=-5.0, b=-3.0, tol=1e-6, max_iter=100)


def test_regula_falsi_invalid_bracket_raises():
    with pytest.raises(ValueError):
        regula_falsi(F_TEXT, a=-5.0, b=-3.0, tol=1e-6, max_iter=100)


def test_bisection_exact_root_at_endpoint():
    # f(x) = x - 2 has an exact root at b=2 -- should short-circuit
    # without iterating at all.
    result = bisection("x - 2", a=0.0, b=2.0, tol=1e-6, max_iter=100)
    assert result.converged
    assert result.solution == pytest.approx(2.0)
    assert result.n_iterations == 0
    assert result.iterations == []


# ----------------------------------------------------------------------
# Additional §6.2 error-path coverage: zero derivative / zero denominator
# ----------------------------------------------------------------------

def test_newton_raphson_zero_derivative_raises():
    # f(x) = x**2 has f'(0) = 0.
    with pytest.raises(ValueError):
        newton_raphson("x**2", x0=0.0, tol=1e-6, max_iter=50)


def test_secant_zero_denominator_raises():
    # x0 == x1 -> f(x0) == f(x1) -> denominator is exactly zero on step 1.
    with pytest.raises(ValueError):
        secant(F_TEXT, x0=1.5, x1=1.5, tol=1e-6, max_iter=50)


def test_fixed_point_warns_on_divergent_rearrangement():
    # g(x) = x**3 - 2 is also a valid rearrangement of x**3 - x - 2 = 0
    # (x = x**3 - 2 <=> x**3 - x - 2 = 0), but |g'(x)| = |3x**2| >> 1
    # near the root, so it should carry a warning rather than raise.
    result = fixed_point("x**3 - 2", x0=1.5, tol=1e-6, max_iter=5)
    assert result.warning is not None
    assert "convergence not guaranteed" in result.warning