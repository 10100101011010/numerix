"""
core/nonlinear.py — nonlinear equation solvers (§6.2).

Five methods: bisection, regula falsi, fixed point, Newton-Raphson,
secant. Every function returns a `MethodResult` (§5) so the UI layer
(added in later phases) renders all five through one shared display
path, and never has its own printing logic.

Convention used throughout this module, per §6.2's "Precondition /
warning" column:
- A precondition that must hold *before* the method can run at all
  (e.g. bisection's `f(a)*f(b) < 0`) raises `ValueError` immediately.
- A condition that arises *during* iteration and makes the method
  unable to continue (e.g. Newton's `f'(x) ≈ 0`) raises `ValueError`
  at that step.
- A condition that only makes convergence *not guaranteed*, but
  doesn't prevent running (e.g. fixed point's `|g'(x0)| >= 1`), is
  attached to the returned `MethodResult.warning` instead — the
  method still runs.
- Simply exhausting `max_iter` without meeting `tol` is normal,
  expected behavior, per §5: it is reported as `converged=False` on
  an otherwise ordinary `MethodResult`, never raised as an error.
"""

from __future__ import annotations

import time

from numerix.utils.parser import parse_function
from numerix.utils.result import MethodResult

_CATEGORY = "Nonlinear Equations"
_ZERO_DERIV_TOL = 1e-12


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


# ----------------------------------------------------------------------
# Bisection
# ----------------------------------------------------------------------

def bisection(f: str, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> MethodResult:
    """Bisection method (§6.2).

    Precondition: `f(a) * f(b) < 0`, else raises `ValueError` before
    running. Stops when `|f(c)| < tol` or the half-interval width
    `(b - a) / 2 < tol`.
    """
    start = time.perf_counter()
    func = parse_function(f, "x")
    a0, b0 = a, b
    fa, fb = func(a), func(b)

    if fa == 0:
        return _trivial_result("Bisection", f, {"a": a0, "b": b0, "tol": tol, "max_iter": max_iter}, a, start)
    if fb == 0:
        return _trivial_result("Bisection", f, {"a": a0, "b": b0, "tol": tol, "max_iter": max_iter}, b, start)
    if fa * fb > 0:
        raise ValueError(
            f"invalid bracket: f(a)*f(b) must be negative "
            f"(got f({a0})={fa:.6g}, f({b0})={fb:.6g})"
        )

    iterations: list[dict] = []
    converged = False
    c = a
    for n in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = func(c)
        half_width = (b - a) / 2
        iterations.append({
            "n": n, "a": a, "b": b, "c": c,
            "f(a)": fa, "f(b)": fb, "f(c)": fc,
            "error": half_width,
        })
        if abs(fc) < tol or half_width < tol:
            converged = True
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return MethodResult(
        method_name="Bisection",
        category=_CATEGORY,
        inputs={"f": f, "a": a0, "b": b0, "tol": tol, "max_iter": max_iter},
        iterations=iterations,
        solution=c,
        approx_error=iterations[-1]["error"] if iterations else 0.0,
        n_iterations=len(iterations),
        converged=converged,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Regula Falsi
# ----------------------------------------------------------------------

def regula_falsi(f: str, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> MethodResult:
    """Regula falsi (false position) method (§6.2).

    Precondition: `f(a) * f(b) < 0`, else raises `ValueError` before
    running. Stops when `|f(c)| < tol`.
    """
    start = time.perf_counter()
    func = parse_function(f, "x")
    a0, b0 = a, b
    fa, fb = func(a), func(b)

    if fa == 0:
        return _trivial_result("Regula Falsi", f, {"a": a0, "b": b0, "tol": tol, "max_iter": max_iter}, a, start)
    if fb == 0:
        return _trivial_result("Regula Falsi", f, {"a": a0, "b": b0, "tol": tol, "max_iter": max_iter}, b, start)
    if fa * fb > 0:
        raise ValueError(
            f"invalid bracket: f(a)*f(b) must be negative "
            f"(got f({a0})={fa:.6g}, f({b0})={fb:.6g})"
        )

    iterations: list[dict] = []
    converged = False
    c = a
    for n in range(1, max_iter + 1):
        c = b - fb * (b - a) / (fb - fa)
        fc = func(c)
        err = abs(fc)
        iterations.append({
            "n": n, "a": a, "b": b, "c": c,
            "f(a)": fa, "f(b)": fb, "f(c)": fc,
            "error": err,
        })
        if err < tol:
            converged = True
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return MethodResult(
        method_name="Regula Falsi",
        category=_CATEGORY,
        inputs={"f": f, "a": a0, "b": b0, "tol": tol, "max_iter": max_iter},
        iterations=iterations,
        solution=c,
        approx_error=iterations[-1]["error"] if iterations else 0.0,
        n_iterations=len(iterations),
        converged=converged,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Fixed Point
# ----------------------------------------------------------------------

def fixed_point(g: str, x0: float, tol: float = 1e-6, max_iter: int = 100) -> MethodResult:
    """Fixed point iteration (§6.2).

    The user supplies `g` already rearranged so that `x = g(x)`.
    Warns (rather than errors) if `|g'(x0)| >= 1`, computed
    symbolically via sympy per §6.1 — convergence is then not
    guaranteed, but the method still runs. Stops when
    `|x_{n+1} - x_n| < tol`.
    """
    start = time.perf_counter()
    func = parse_function(g, "x")

    warning: str | None = None
    try:
        slope = func.derivative("x")(x0)
        if abs(slope) >= 1:
            warning = f"|g'(x0)| = {abs(slope):.6g} >= 1 — convergence not guaranteed"
    except ValueError:
        # g'(x0) isn't evaluable (e.g. domain issue right at x0) --
        # skip the precondition check rather than blocking the run.
        pass

    iterations: list[dict] = []
    converged = False
    x_curr = x0
    for n in range(1, max_iter + 1):
        x_next = func(x_curr)
        err = abs(x_next - x_curr)
        iterations.append({"n": n, "x_n": x_curr, "x_next": x_next, "error": err})
        x_curr = x_next
        if err < tol:
            converged = True
            break

    return MethodResult(
        method_name="Fixed Point",
        category=_CATEGORY,
        inputs={"g": g, "x0": x0, "tol": tol, "max_iter": max_iter},
        iterations=iterations,
        solution=x_curr,
        approx_error=iterations[-1]["error"] if iterations else 0.0,
        n_iterations=len(iterations),
        converged=converged,
        warning=warning,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Newton-Raphson
# ----------------------------------------------------------------------

def newton_raphson(f: str, x0: float, tol: float = 1e-6, max_iter: int = 100) -> MethodResult:
    """Newton-Raphson method (§6.2).

    `f'(x)` is derived automatically via `sympy.diff` (§6.1) — the
    user never types a derivative. Raises `ValueError` if `f'(x_n)`
    is ever ≈ 0. Stops when `|x_{n+1} - x_n| < tol`.
    """
    start = time.perf_counter()
    func = parse_function(f, "x")
    fprime = func.derivative("x")

    iterations: list[dict] = []
    converged = False
    x_curr = x0
    for n in range(1, max_iter + 1):
        fx = func(x_curr)
        dfx = fprime(x_curr)
        if abs(dfx) < _ZERO_DERIV_TOL:
            raise ValueError(
                f"f'(x) ≈ 0 at x = {x_curr:.6g} (iteration {n}) — "
                f"Newton-Raphson cannot continue"
            )
        x_next = x_curr - fx / dfx
        err = abs(x_next - x_curr)
        iterations.append({
            "n": n, "x_n": x_curr, "f(x_n)": fx, "f'(x_n)": dfx,
            "x_next": x_next, "error": err,
        })
        x_curr = x_next
        if err < tol:
            converged = True
            break

    return MethodResult(
        method_name="Newton-Raphson",
        category=_CATEGORY,
        inputs={"f": f, "x0": x0, "tol": tol, "max_iter": max_iter},
        iterations=iterations,
        solution=x_curr,
        approx_error=iterations[-1]["error"] if iterations else 0.0,
        n_iterations=len(iterations),
        converged=converged,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Secant
# ----------------------------------------------------------------------

def secant(f: str, x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100) -> MethodResult:
    """Secant method (§6.2).

    Raises `ValueError` if the denominator `f(x_n) - f(x_{n-1})` is
    ever ≈ 0. Stops when `|x_{n+1} - x_n| < tol`.
    """
    start = time.perf_counter()
    func = parse_function(f, "x")

    iterations: list[dict] = []
    converged = False
    x_prev, x_curr = x0, x1
    f_prev = func(x_prev)
    for n in range(1, max_iter + 1):
        f_curr = func(x_curr)
        denom = f_curr - f_prev
        if abs(denom) < _ZERO_DERIV_TOL:
            raise ValueError(
                f"denominator ≈ 0 at iteration {n} "
                f"(f(x_n) - f(x_(n-1)) ≈ 0) — Secant method cannot continue"
            )
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        err = abs(x_next - x_curr)
        iterations.append({
            "n": n, "x_prev": x_prev, "x_curr": x_curr,
            "f(x_prev)": f_prev, "f(x_curr)": f_curr,
            "x_next": x_next, "error": err,
        })
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_next
        if err < tol:
            converged = True
            break

    return MethodResult(
        method_name="Secant",
        category=_CATEGORY,
        inputs={"f": f, "x0": x0, "x1": x1, "tol": tol, "max_iter": max_iter},
        iterations=iterations,
        solution=x_curr,
        approx_error=iterations[-1]["error"] if iterations else 0.0,
        n_iterations=len(iterations),
        converged=converged,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Shared helper
# ----------------------------------------------------------------------

def _trivial_result(method_name: str, f_text: str, inputs: dict, root: float, start: float) -> MethodResult:
    """Build the zero-iteration MethodResult for an exact root found at a bracket endpoint."""
    return MethodResult(
        method_name=method_name,
        category=_CATEGORY,
        inputs={"f": f_text, **inputs},
        iterations=[],
        solution=root,
        approx_error=0.0,
        n_iterations=0,
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )