"""
tests/test_linear_systems.py — tests for numerix.core.linear_systems (§6.3, §11).
"""

import pytest

from numerix.core.linear_systems import (
    gauss_elimination,
    gauss_jordan,
    gauss_seidel,
    jacobi,
    lu_decomposition,
    matrix_inverse,
)

# Classic hand-verified 3x3 system (§11): x=2, y=3, z=-1
A_DIRECT = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
B_DIRECT = [8.0, -11.0, -3.0]
X_DIRECT = [2.0, 3.0, -1.0]

# Strictly diagonally dominant 3x3 system for the iterative methods: x=y=z=1
A_DOMINANT = [[10.0, 1.0, 1.0], [2.0, 10.0, 1.0], [2.0, 1.0, 10.0]]
B_DOMINANT = [12.0, 13.0, 13.0]
X_DOMINANT = [1.0, 1.0, 1.0]

# Singular matrix (row 2 = 2 * row 1)
A_SINGULAR = [[1.0, 2.0], [2.0, 4.0]]
B_SINGULAR = [3.0, 6.0]

TOL = 1e-4


def _close(vec, expected, tol=TOL) -> bool:
    return all(abs(v - e) < tol for v, e in zip(vec, expected))


# ----------------------------------------------------------------------
# Known-answer tests (§11: "a small well-conditioned linear system
# with a hand-verified solution")
# ----------------------------------------------------------------------

def test_gauss_elimination_known_solution():
    result = gauss_elimination(A_DIRECT, B_DIRECT)
    assert result.converged
    assert _close(result.solution, X_DIRECT)
    assert result.method_name == "Gaussian Elimination"
    assert result.category == "Linear Systems"


def test_gauss_jordan_known_solution():
    result = gauss_jordan(A_DIRECT, B_DIRECT)
    assert result.converged
    assert _close(result.solution, X_DIRECT)


def test_lu_decomposition_known_solution():
    result = lu_decomposition(A_DIRECT, B_DIRECT)
    assert result.converged
    assert _close(result.solution["x"], X_DIRECT)
    # §6.3: "show L and U in the result" -- reconstruct A = L @ U and check it matches.
    L, U = result.solution["L"], result.solution["U"]
    n = len(A_DIRECT)
    reconstructed = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            assert abs(reconstructed[i][j] - A_DIRECT[i][j]) < TOL


def test_matrix_inverse_known_solution():
    result = matrix_inverse(A_DIRECT)
    inv = result.solution
    n = len(A_DIRECT)
    product = [[sum(A_DIRECT[i][k] * inv[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            assert abs(product[i][j] - expected) < TOL


def test_jacobi_known_solution():
    result = jacobi(A_DOMINANT, B_DOMINANT, x0=[0.0, 0.0, 0.0], tol=1e-8, max_iter=200)
    assert result.converged
    assert _close(result.solution, X_DOMINANT)
    assert result.warning is None  # A_DOMINANT is diagonally dominant


def test_gauss_seidel_known_solution():
    result = gauss_seidel(A_DOMINANT, B_DOMINANT, x0=[0.0, 0.0, 0.0], tol=1e-8, max_iter=200)
    assert result.converged
    assert _close(result.solution, X_DOMINANT)
    assert result.warning is None


def test_gauss_seidel_converges_no_slower_than_jacobi():
    # Standard property: Gauss-Seidel uses updated values within the
    # sweep, so it should need no more iterations than Jacobi here.
    jacobi_result = jacobi(A_DOMINANT, B_DOMINANT, x0=[0.0, 0.0, 0.0], tol=1e-8, max_iter=200)
    gs_result = gauss_seidel(A_DOMINANT, B_DOMINANT, x0=[0.0, 0.0, 0.0], tol=1e-8, max_iter=200)
    assert gs_result.n_iterations <= jacobi_result.n_iterations


# ----------------------------------------------------------------------
# §11 edge case: singular matrix
# ----------------------------------------------------------------------

def test_gauss_elimination_singular_raises():
    with pytest.raises(ValueError):
        gauss_elimination(A_SINGULAR, B_SINGULAR)


def test_gauss_jordan_singular_raises():
    with pytest.raises(ValueError):
        gauss_jordan(A_SINGULAR, B_SINGULAR)


def test_lu_decomposition_singular_raises():
    with pytest.raises(ValueError):
        lu_decomposition(A_SINGULAR, B_SINGULAR)


def test_matrix_inverse_singular_raises():
    with pytest.raises(ValueError):
        matrix_inverse(A_SINGULAR)


# ----------------------------------------------------------------------
# §11 edge case: non-diagonally-dominant matrix (should still run, warn)
# ----------------------------------------------------------------------

def test_jacobi_warns_on_non_dominant_matrix():
    # A_DIRECT is not diagonally dominant (row 1: |-1| < |-3| + |2|).
    result = jacobi(A_DIRECT, B_DIRECT, x0=[0.0, 0.0, 0.0], tol=1e-6, max_iter=10)
    assert result.warning is not None
    assert "diagonally dominant" in result.warning


def test_gauss_seidel_warns_on_non_dominant_matrix():
    result = gauss_seidel(A_DIRECT, B_DIRECT, x0=[0.0, 0.0, 0.0], tol=1e-6, max_iter=10)
    assert result.warning is not None
    assert "diagonally dominant" in result.warning


# ----------------------------------------------------------------------
# §8: wrong dimensions / non-square matrices must not raw-crash
# ----------------------------------------------------------------------

def test_gauss_elimination_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        gauss_elimination([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0, 3.0])


def test_gauss_elimination_non_square_raises():
    with pytest.raises(ValueError):
        gauss_elimination([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])


def test_jacobi_x0_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        jacobi(A_DOMINANT, B_DOMINANT, x0=[0.0, 0.0], tol=1e-6, max_iter=10)


def test_matrix_inverse_non_square_raises():
    with pytest.raises(ValueError):
        matrix_inverse([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# ----------------------------------------------------------------------
# §11 edge case: non-convergence (max_iter hit without meeting tol)
# ----------------------------------------------------------------------

def test_jacobi_non_convergence_reports_gracefully():
    result = jacobi(A_DOMINANT, B_DOMINANT, x0=[0.0, 0.0, 0.0], tol=1e-15, max_iter=2)
    assert result.converged is False
    assert result.n_iterations == 2


def test_gauss_seidel_non_convergence_reports_gracefully():
    result = gauss_seidel(A_DOMINANT, B_DOMINANT, x0=[0.0, 0.0, 0.0], tol=1e-15, max_iter=2)
    assert result.converged is False
    assert result.n_iterations == 2