"""
core/interpolation.py — interpolation methods (§6.4).

Seven methods: linear, quadratic, cubic, Lagrange, Newton divided
difference, Newton-Gregory forward, Newton-Gregory backward. All
hand-implemented on plain Python lists — no library interpolators.
Every function returns a `MethodResult` (§5).

Per §6.4: `iterations` holds the difference/divided-difference table
itself (or, for linear/quadratic/cubic/Lagrange, the Lagrange basis
term for each point) rather than a convergence loop, so the user can
see how the polynomial was built.

Linear, quadratic, and cubic interpolation are all just the general
Lagrange formula restricted to exactly 2, 3, or 4 points respectively
— mathematically identical, so implementing them as thin wrappers
around one shared Lagrange routine keeps them exact and avoids three
near-duplicate formulas.
"""

from __future__ import annotations

import math
import time

from numerix.utils.result import MethodResult

_CATEGORY = "Interpolation"


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_points(
    xs: list[float], ys: list[float], *, expected_n: int | None = None, min_n: int = 2, method_label: str = ""
) -> int:
    if len(xs) != len(ys):
        raise ValueError(f"x and y must have the same number of points (got {len(xs)} x-values, {len(ys)} y-values)")
    n = len(xs)
    if expected_n is not None and n != expected_n:
        raise ValueError(f"{method_label} requires exactly {expected_n} points, got {n}")
    if n < min_n:
        raise ValueError(f"{method_label} requires at least {min_n} points, got {n}")
    if len(set(xs)) != n:
        raise ValueError("x-values must be distinct (duplicate x found)")
    return n


def _check_equally_spaced(xs: list[float], method_label: str, tol: float = 1e-9) -> float:
    h = xs[1] - xs[0]
    if abs(h) < 1e-12:
        raise ValueError(f"{method_label} requires distinct, increasing x-values (step size is zero)")
    for i in range(1, len(xs) - 1):
        if abs((xs[i + 1] - xs[i]) - h) > tol * max(1.0, abs(h)):
            raise ValueError(f"{method_label} requires equally spaced x-values (step size must be constant)")
    return h


# ----------------------------------------------------------------------
# Shared Lagrange routine (backs linear, quadratic, cubic, and Lagrange itself)
# ----------------------------------------------------------------------

def _lagrange_value_and_rows(xs: list[float], ys: list[float], x_target: float) -> tuple[float, list[dict]]:
    n = len(xs)
    rows: list[dict] = []
    total = 0.0
    for i in range(n):
        basis = 1.0
        for j in range(n):
            if j != i:
                basis *= (x_target - xs[j]) / (xs[i] - xs[j])
        term = ys[i] * basis
        total += term
        rows.append({"i": i + 1, "x_i": xs[i], "y_i": ys[i], "L_i(x_target)": basis, "term": term})
    return total, rows


def _interpolate_via_lagrange(
    method_name: str, xs: list[float], ys: list[float], x_target: float, *, expected_n: int | None, min_n: int = 2
) -> MethodResult:
    start = time.perf_counter()
    _validate_points(xs, ys, expected_n=expected_n, min_n=min_n, method_label=method_name)
    xs0, ys0 = list(xs), list(ys)
    value, rows = _lagrange_value_and_rows(xs, ys, x_target)
    return MethodResult(
        method_name=method_name,
        category=_CATEGORY,
        inputs={"xs": xs0, "ys": ys0, "x_target": x_target},
        iterations=rows,
        solution=value,
        approx_error=None,
        n_iterations=len(rows),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


def linear(xs: list[float], ys: list[float], x_target: float) -> MethodResult:
    """Straight-line formula between exactly 2 points (§6.4)."""
    return _interpolate_via_lagrange("Linear Interpolation", xs, ys, x_target, expected_n=2)


def quadratic(xs: list[float], ys: list[float], x_target: float) -> MethodResult:
    """Degree-2 polynomial fit through exactly 3 points (§6.4)."""
    return _interpolate_via_lagrange("Quadratic Interpolation", xs, ys, x_target, expected_n=3)


def cubic(xs: list[float], ys: list[float], x_target: float) -> MethodResult:
    """Degree-3 polynomial fit through exactly 4 points (§6.4)."""
    return _interpolate_via_lagrange("Cubic Interpolation", xs, ys, x_target, expected_n=4)


def lagrange(xs: list[float], ys: list[float], x_target: float) -> MethodResult:
    """Full Lagrange basis-polynomial formula for general n (§6.4)."""
    return _interpolate_via_lagrange("Lagrange Interpolation", xs, ys, x_target, expected_n=None, min_n=2)


# ----------------------------------------------------------------------
# Newton Divided Difference (unequal spacing allowed)
# ----------------------------------------------------------------------

def newton_divided_diff(xs: list[float], ys: list[float], x_target: float) -> MethodResult:
    """Newton's divided-difference form; works for unequally spaced points (§6.4)."""
    start = time.perf_counter()
    n = _validate_points(xs, ys, min_n=2, method_label="Newton Divided Difference")
    xs0, ys0 = list(xs), list(ys)

    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (xs[i + j] - xs[i])

    value = table[0][0]
    product_term = 1.0
    for j in range(1, n):
        product_term *= (x_target - xs[j - 1])
        value += table[0][j] * product_term

    rows = []
    for i in range(n):
        row = {"x_i": xs[i]}
        for j in range(n - i):
            row[f"order_{j}"] = table[i][j]
        rows.append(row)

    return MethodResult(
        method_name="Newton Divided Difference",
        category=_CATEGORY,
        inputs={"xs": xs0, "ys": ys0, "x_target": x_target},
        iterations=rows,
        solution=value,
        approx_error=None,
        n_iterations=len(rows),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Newton-Gregory Forward (equally spaced, x_target near the start)
# ----------------------------------------------------------------------

def newton_gregory_forward(xs: list[float], ys: list[float], x_target: float) -> MethodResult:
    """Forward-difference table form; requires equally spaced points (§6.4)."""
    start = time.perf_counter()
    n = _validate_points(xs, ys, min_n=2, method_label="Newton-Gregory Forward")
    xs0, ys0 = list(xs), list(ys)
    h = _check_equally_spaced(xs, "Newton-Gregory Forward")

    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    p = (x_target - xs[0]) / h
    value = table[0][0]
    term = 1.0
    for k in range(1, n):
        term *= (p - (k - 1))
        value += term / math.factorial(k) * table[0][k]

    rows = []
    for i in range(n):
        row = {"x_i": xs[i]}
        for j in range(n - i):
            row[f"order_{j}"] = table[i][j]
        rows.append(row)

    return MethodResult(
        method_name="Newton-Gregory Forward",
        category=_CATEGORY,
        inputs={"xs": xs0, "ys": ys0, "x_target": x_target},
        iterations=rows,
        solution=value,
        approx_error=None,
        n_iterations=len(rows),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Newton-Gregory Backward (equally spaced, x_target near the end)
# ----------------------------------------------------------------------

def newton_gregory_backward(xs: list[float], ys: list[float], x_target: float) -> MethodResult:
    """Backward-difference table form; requires equally spaced points (§6.4)."""
    start = time.perf_counter()
    n = _validate_points(xs, ys, min_n=2, method_label="Newton-Gregory Backward")
    xs0, ys0 = list(xs), list(ys)
    h = _check_equally_spaced(xs, "Newton-Gregory Backward")

    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]
    for k in range(1, n):
        for i in range(k, n):
            table[i][k] = table[i][k - 1] - table[i - 1][k - 1]

    p = (x_target - xs[n - 1]) / h
    value = table[n - 1][0]
    term = 1.0
    for k in range(1, n):
        term *= (p + (k - 1))
        value += term / math.factorial(k) * table[n - 1][k]

    rows = []
    for i in range(n):
        row = {"x_i": xs[i]}
        for j in range(i + 1):
            row[f"order_{j}"] = table[i][j]
        rows.append(row)

    return MethodResult(
        method_name="Newton-Gregory Backward",
        category=_CATEGORY,
        inputs={"xs": xs0, "ys": ys0, "x_target": x_target},
        iterations=rows,
        solution=value,
        approx_error=None,
        n_iterations=len(rows),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )