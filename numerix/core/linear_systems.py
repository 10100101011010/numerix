"""
core/linear_systems.py — linear systems solvers (§6.3).

Six methods: Gaussian elimination, Gauss-Jordan, matrix inverse, LU
decomposition (Doolittle), Jacobi, Gauss-Seidel. All hand-implemented
on plain Python lists of lists — no `numpy.linalg` or other library
solver, per §6 constraint. Every function returns a `MethodResult`
(§5).

Convention (matches `core/nonlinear.py`):
- A hard precondition failure (singular matrix, dimension mismatch,
  non-square matrix, zero diagonal entry) raises `ValueError`.
- A condition that only makes convergence *not guaranteed* (matrix
  not diagonally dominant, for Jacobi/Gauss-Seidel) is attached to
  `MethodResult.warning` instead — the method still runs.
- Exhausting `max_iter` without meeting `tol` is normal, expected
  behavior: reported as `converged=False`, never raised.
"""

from __future__ import annotations

import time

from numerix.utils.result import MethodResult

_CATEGORY = "Linear Systems"
_EPS = 1e-12


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_matrix(A: list[list[float]]) -> int:
    """Validate A is a non-empty square matrix; return its size n."""
    if not A:
        raise ValueError("matrix A cannot be empty")
    n = len(A)
    for i, row in enumerate(A):
        if len(row) != n:
            raise ValueError(
                f"matrix A must be square ({n}x{n}) — row {i} has length {len(row)}, expected {n}"
            )
    return n


def _validate_system(A: list[list[float]], b: list[float]) -> int:
    n = _validate_matrix(A)
    if len(b) != n:
        raise ValueError(f"vector b must have length {n} to match A ({n}x{n}) — got length {len(b)}")
    return n


def _is_diagonally_dominant(A: list[list[float]]) -> bool:
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag < off_diag_sum:
            return False
    return True


def _format_matrix(M: list[list[float]], aug_col: int | None = None) -> str:
    """Render a matrix as a multi-line string for iteration snapshots.

    If `aug_col` is given, columns from that index on are shown after
    a ' | ' separator (for an augmented matrix like `[A | b]`).
    """
    lines = []
    for row in M:
        if aug_col is not None:
            main = "  ".join(f"{v:8.4g}" for v in row[:aug_col])
            aug = "  ".join(f"{v:8.4g}" for v in row[aug_col:])
            lines.append(f"{main} | {aug}")
        else:
            lines.append("  ".join(f"{v:8.4g}" for v in row))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Gaussian Elimination
# ----------------------------------------------------------------------

def gauss_elimination(A: list[list[float]], b: list[float]) -> MethodResult:
    """Gaussian elimination with partial pivoting, then back-substitution (§6.3)."""
    start = time.perf_counter()
    n = _validate_system(A, b)
    A0 = [row[:] for row in A]
    b0 = list(b)
    M = [list(row) + [bi] for row, bi in zip(A, b)]

    iterations: list[dict] = []
    for k in range(n - 1):
        pivot_row = max(range(k, n), key=lambda r: abs(M[r][k]))
        if abs(M[pivot_row][k]) < _EPS:
            raise ValueError(f"matrix is singular — no valid pivot in column {k} (Gaussian elimination cannot continue)")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
        iterations.append({"step": k + 1, "pivot_row": pivot_row + 1, "matrix": _format_matrix(M, aug_col=n)})

    if abs(M[n - 1][n - 1]) < _EPS:
        raise ValueError("matrix is singular — zero pivot at the final row (Gaussian elimination cannot continue)")

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]

    return MethodResult(
        method_name="Gaussian Elimination",
        category=_CATEGORY,
        inputs={"A": A0, "b": b0},
        iterations=iterations,
        solution=x,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Gauss-Jordan
# ----------------------------------------------------------------------

def gauss_jordan(A: list[list[float]], b: list[float]) -> MethodResult:
    """Gauss-Jordan elimination straight to RREF, no separate back-substitution (§6.3)."""
    start = time.perf_counter()
    n = _validate_system(A, b)
    A0 = [row[:] for row in A]
    b0 = list(b)
    M = [list(row) + [bi] for row, bi in zip(A, b)]

    iterations: list[dict] = []
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(M[r][k]))
        if abs(M[pivot_row][k]) < _EPS:
            raise ValueError(f"matrix is singular — no valid pivot in column {k} (Gauss-Jordan cannot continue)")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        pivot_val = M[k][k]
        M[k] = [v / pivot_val for v in M[k]]
        for i in range(n):
            if i != k:
                factor = M[i][k]
                M[i] = [M[i][j] - factor * M[k][j] for j in range(n + 1)]
        iterations.append({"step": k + 1, "pivot_row": pivot_row + 1, "matrix": _format_matrix(M, aug_col=n)})

    x = [M[i][n] for i in range(n)]

    return MethodResult(
        method_name="Gauss-Jordan",
        category=_CATEGORY,
        inputs={"A": A0, "b": b0},
        iterations=iterations,
        solution=x,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Matrix Inverse
# ----------------------------------------------------------------------

def matrix_inverse(A: list[list[float]], b: list[float] | None = None) -> MethodResult:
    """Gauss-Jordan on `[A | I]` until the left side is `I` (§6.3).

    `b` is optional. When supplied, also computes `x = A\u207b\u00b9\u00b7b`
    using the freshly-computed inverse and reports it as a clearly
    labeled secondary result alongside the inverse — the inverse
    matrix itself stays the primary output either way.
    """
    start = time.perf_counter()
    n = _validate_matrix(A)
    if b is not None and len(b) != n:
        raise ValueError(f"vector b must have length {n} to match A ({n}x{n}) — got length {len(b)}")
    A0 = [row[:] for row in A]
    b0 = list(b) if b is not None else None
    M = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

    iterations: list[dict] = []
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(M[r][k]))
        if abs(M[pivot_row][k]) < _EPS:
            raise ValueError("matrix is singular (det(A) \u2248 0) \u2014 inverse does not exist")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        pivot_val = M[k][k]
        M[k] = [v / pivot_val for v in M[k]]
        for i in range(n):
            if i != k:
                factor = M[i][k]
                M[i] = [M[i][j] - factor * M[k][j] for j in range(2 * n)]
        iterations.append({"step": k + 1, "pivot_row": pivot_row + 1, "matrix": _format_matrix(M, aug_col=n)})

    inverse = [row[n:] for row in M]

    inputs: dict[str, object] = {"A": A0}
    if b0 is not None:
        inputs["b"] = b0
        x = [sum(inverse[i][j] * b0[j] for j in range(n)) for i in range(n)]
        solution: object = {"Inverse (A\u207b\u00b9)": inverse, "x = A\u207b\u00b9\u00b7b": x}
    else:
        solution = inverse

    return MethodResult(
        method_name="Matrix Inverse",
        category=_CATEGORY,
        inputs=inputs,
        iterations=iterations,
        solution=solution,
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# LU Decomposition (Doolittle)
# ----------------------------------------------------------------------

def lu_decomposition(A: list[list[float]], b: list[float]) -> MethodResult:
    """Doolittle LU decomposition (L unit-diagonal, U upper), then forward/back substitution (§6.3)."""
    start = time.perf_counter()
    n = _validate_system(A, b)
    A0 = [row[:] for row in A]
    b0 = list(b)

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    iterations: list[dict] = []
    for k in range(n):
        for j in range(k, n):
            U[k][j] = A[k][j] - sum(L[k][m] * U[m][j] for m in range(k))
        if abs(U[k][k]) < _EPS:
            raise ValueError(f"matrix is singular — zero pivot U[{k}][{k}] (LU decomposition cannot continue)")
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = (A[i][k] - sum(L[i][m] * U[m][k] for m in range(k))) / U[k][k]
        iterations.append({"step": k + 1, "L": _format_matrix(L), "U": _format_matrix(U)})

    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return MethodResult(
        method_name="LU Decomposition",
        category=_CATEGORY,
        inputs={"A": A0, "b": b0},
        iterations=iterations,
        solution={"x": x, "L": L, "U": U},
        approx_error=None,
        n_iterations=len(iterations),
        converged=True,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Jacobi
# ----------------------------------------------------------------------

def jacobi(
    A: list[list[float]], b: list[float], x0: list[float], tol: float = 1e-6, max_iter: int = 100
) -> MethodResult:
    """Jacobi iteration: every x_i updated from the *previous* full iterate (§6.3)."""
    start = time.perf_counter()
    n = _validate_system(A, b)
    if len(x0) != n:
        raise ValueError(f"initial guess x0 must have length {n} to match A ({n}x{n}) — got length {len(x0)}")
    for i in range(n):
        if abs(A[i][i]) < _EPS:
            raise ValueError(f"zero (or near-zero) diagonal entry at row {i} \u2014 Jacobi requires nonzero diagonal entries")

    A0, b0, x0_orig = [row[:] for row in A], list(b), list(x0)
    warning = None
    if not _is_diagonally_dominant(A):
        warning = "matrix A is not diagonally dominant \u2014 convergence not guaranteed"

    x = list(x0)
    iterations: list[dict] = []
    converged = False
    for n_iter in range(1, max_iter + 1):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        err = max(abs(x_new[i] - x[i]) for i in range(n))
        row = {"n": n_iter, **{f"x{i + 1}": x_new[i] for i in range(n)}, "error": err}
        iterations.append(row)
        x = x_new
        if err < tol:
            converged = True
            break

    return MethodResult(
        method_name="Jacobi",
        category=_CATEGORY,
        inputs={"A": A0, "b": b0, "x0": x0_orig, "tol": tol, "max_iter": max_iter},
        iterations=iterations,
        solution=x,
        approx_error=iterations[-1]["error"] if iterations else 0.0,
        n_iterations=len(iterations),
        converged=converged,
        warning=warning,
        exec_time_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Gauss-Seidel
# ----------------------------------------------------------------------

def gauss_seidel(
    A: list[list[float]], b: list[float], x0: list[float], tol: float = 1e-6, max_iter: int = 100
) -> MethodResult:
    """Gauss-Seidel iteration: uses updated x_i values within the same sweep (§6.3)."""
    start = time.perf_counter()
    n = _validate_system(A, b)
    if len(x0) != n:
        raise ValueError(f"initial guess x0 must have length {n} to match A ({n}x{n}) — got length {len(x0)}")
    for i in range(n):
        if abs(A[i][i]) < _EPS:
            raise ValueError(f"zero (or near-zero) diagonal entry at row {i} \u2014 Gauss-Seidel requires nonzero diagonal entries")

    A0, b0, x0_orig = [row[:] for row in A], list(b), list(x0)
    warning = None
    if not _is_diagonally_dominant(A):
        warning = "matrix A is not diagonally dominant \u2014 convergence not guaranteed"

    x = list(x0)
    iterations: list[dict] = []
    converged = False
    for n_iter in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        err = max(abs(x[i] - x_old[i]) for i in range(n))
        row = {"n": n_iter, **{f"x{i + 1}": x[i] for i in range(n)}, "error": err}
        iterations.append(row)
        if err < tol:
            converged = True
            break

    return MethodResult(
        method_name="Gauss-Seidel",
        category=_CATEGORY,
        inputs={"A": A0, "b": b0, "x0": x0_orig, "tol": tol, "max_iter": max_iter},
        iterations=iterations,
        solution=x,
        approx_error=iterations[-1]["error"] if iterations else 0.0,
        n_iterations=len(iterations),
        converged=converged,
        warning=warning,
        exec_time_ms=_elapsed_ms(start),
    )