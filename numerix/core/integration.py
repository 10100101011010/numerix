"""
core/integration.py — numerical integration methods (§6.5).

Five methods: rectangle rule (left/right), midpoint rule, trapezoidal
rule, Simpson's 1/3 rule, Simpson's 3/8 rule. All hand-implemented —
no library quadrature. Every function returns a `MethodResult` (§5).

Per §6.5: these are non-iterative in the "convergence loop" sense,
but `iterations` still holds one row per subinterval/node
(`i, x_i, f(x_i), weight`) so the user sees the full computation, not
just the final number.
"""

from __future__ import annotations

import time

from numerix.utils.parser import parse_function
from numerix.utils.result import MethodResult

_CATEGORY = "Numerical Integration"


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _validate_bounds(a: float, b: float) -> None:
    if b <= a:
        raise ValueError(f"b must be greater than a (got a={a}, b={b})")


def _validate_n(n: int, min_n: int = 1) -> int:
    n_int = int(n)
    if n_int != n or n_int < min_n:
        raise ValueError(f"n must be an integer >= {min_n}, got {n}")
    return n_int


# ----------------------------------------------------------------------
# Rectangle Rule
# ----------------------------------------------------------------------

def rectangle_rule(f: str, a: float, b: float, n: int, side: str = "left") -> MethodResult:
    """Left- or right-endpoint rectangle rule: `h * Σ f(x_i)` (§6.5)."""
    start = time.perf_counter()
    _validate_bounds(a, b)
    n = _validate_n(n)
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    func = parse_function(f, "x")
    h = (b - a) / n

    iterations: list[dict] = []
    total = 0.0
    for i in range(n):
        x_i = a + i * h if side == "left" else a + (i + 1) * h
        f_val = func(x_i)
        total += f_val
        iterations.append({"i": i + 1, "x_i": x_i, "f(x_i)": f_val, "weight": h})
    value = h * total

    return MethodResult(
        method_name="Rectangle Rule",
        category=_CATEGORY,
        inputs={"f": f, "a": a, "b": b, "n": n, "side": side},
        iterations=iterations,
        solution=value,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Midpoint Rule
# ----------------------------------------------------------------------

def midpoint_rule(f: str, a: float, b: float, n: int) -> MethodResult:
    """Midpoint rule: `h * Σ f(midpoint_i)` (§6.5)."""
    start = time.perf_counter()
    _validate_bounds(a, b)
    n = _validate_n(n)
    func = parse_function(f, "x")
    h = (b - a) / n

    iterations: list[dict] = []
    total = 0.0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        f_val = func(x_mid)
        total += f_val
        iterations.append({"i": i + 1, "x_i": x_mid, "f(x_i)": f_val, "weight": h})
    value = h * total

    return MethodResult(
        method_name="Midpoint Rule",
        category=_CATEGORY,
        inputs={"f": f, "a": a, "b": b, "n": n},
        iterations=iterations,
        solution=value,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Trapezoidal Rule
# ----------------------------------------------------------------------

def trapezoidal_rule(f: str, a: float, b: float, n: int) -> MethodResult:
    """Trapezoidal rule: `h/2 * [f(a) + 2*Σf(x_i) + f(b)]` (§6.5)."""
    start = time.perf_counter()
    _validate_bounds(a, b)
    n = _validate_n(n)
    func = parse_function(f, "x")
    h = (b - a) / n

    xs = [a + i * h for i in range(n + 1)]
    f_vals = [func(x) for x in xs]
    interior_sum = sum(f_vals[1:n])
    value = h / 2 * (f_vals[0] + 2 * interior_sum + f_vals[n])

    iterations = [
        {"i": i, "x_i": xs[i], "f(x_i)": f_vals[i], "weight": (h / 2 if i in (0, n) else h)}
        for i in range(n + 1)
    ]

    return MethodResult(
        method_name="Trapezoidal Rule",
        category=_CATEGORY,
        inputs={"f": f, "a": a, "b": b, "n": n},
        iterations=iterations,
        solution=value,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Simpson's 1/3 Rule
# ----------------------------------------------------------------------

def simpson_1_3(f: str, a: float, b: float, n: int) -> MethodResult:
    """Simpson's 1/3 rule: `h/3 * [f(a) + 4*Σ_odd + 2*Σ_even + f(b)]`, n even (§6.5)."""
    start = time.perf_counter()
    _validate_bounds(a, b)
    n = _validate_n(n, min_n=2)
    if n % 2 != 0:
        raise ValueError(f"Simpson's 1/3 rule requires n to be even, got n={n}")
    func = parse_function(f, "x")
    h = (b - a) / n

    xs = [a + i * h for i in range(n + 1)]
    f_vals = [func(x) for x in xs]
    odd_sum = sum(f_vals[i] for i in range(1, n, 2))
    even_sum = sum(f_vals[i] for i in range(2, n, 2))
    value = h / 3 * (f_vals[0] + 4 * odd_sum + 2 * even_sum + f_vals[n])

    iterations = []
    for i in range(n + 1):
        coeff = 1 if i in (0, n) else (4 if i % 2 == 1 else 2)
        iterations.append({"i": i, "x_i": xs[i], "f(x_i)": f_vals[i], "weight": h / 3 * coeff})

    return MethodResult(
        method_name="Simpson's 1/3 Rule",
        category=_CATEGORY,
        inputs={"f": f, "a": a, "b": b, "n": n},
        iterations=iterations,
        solution=value,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Simpson's 3/8 Rule
# ----------------------------------------------------------------------

def simpson_3_8(f: str, a: float, b: float, n: int) -> MethodResult:
    """Simpson's 3/8 rule: `3h/8 * [f(a) + 3*Σ(non-mult-3) + 2*Σ(mult-3) + f(b)]`, n divisible by 3 (§6.5)."""
    start = time.perf_counter()
    _validate_bounds(a, b)
    n = _validate_n(n, min_n=3)
    if n % 3 != 0:
        raise ValueError(f"Simpson's 3/8 rule requires n to be divisible by 3, got n={n}")
    func = parse_function(f, "x")
    h = (b - a) / n

    xs = [a + i * h for i in range(n + 1)]
    f_vals = [func(x) for x in xs]
    non_mult3_sum = sum(f_vals[i] for i in range(1, n) if i % 3 != 0)
    mult3_sum = sum(f_vals[i] for i in range(1, n) if i % 3 == 0)
    value = 3 * h / 8 * (f_vals[0] + 3 * non_mult3_sum + 2 * mult3_sum + f_vals[n])

    iterations = []
    for i in range(n + 1):
        coeff = 1 if i in (0, n) else (2 if i % 3 == 0 else 3)
        iterations.append({"i": i, "x_i": xs[i], "f(x_i)": f_vals[i], "weight": 3 * h / 8 * coeff})

    return MethodResult(
        method_name="Simpson's 3/8 Rule",
        category=_CATEGORY,
        inputs={"f": f, "a": a, "b": b, "n": n},
        iterations=iterations,
        solution=value,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )