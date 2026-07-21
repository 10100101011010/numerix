"""
ui/prompts.py — collects and validates user input per method (§7).

Each nonlinear method has a small, declarative list of `Field`s (in
the same order as Section 6.2's "Required inputs" column).
`collect_inputs` walks that list with `questionary.text`, validating
each answer before moving on, and returns a plain dict ready to
`**kwargs` straight into the matching `core.nonlinear` function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Field:
    name: str              # kwarg name expected by the core function
    label: str              # text shown to the user
    kind: str                # "text" | "float" | "int"
    default: Any = None


def _validate_text(text: str) -> "bool | str":
    if not text.strip():
        return "This field is required."
    return True


def _validate_float(text: str) -> "bool | str":
    if not text.strip():
        return "This field is required."
    try:
        float(text)
    except ValueError:
        return f"'{text}' is not a valid number."
    return True


def _validate_int(text: str) -> "bool | str":
    if not text.strip():
        return "This field is required."
    try:
        int(text)
    except ValueError:
        return f"'{text}' is not a valid integer."
    return True


_VALIDATORS: dict[str, Callable[[str], "bool | str"]] = {
    "text": _validate_text,
    "float": _validate_float,
    "int": _validate_int,
}

_CASTERS: dict[str, Callable[[str], Any]] = {
    "text": str,
    "float": float,
    "int": int,
}


def collect_inputs(fields: list[Field]) -> Optional[dict[str, Any]]:
    """Prompt for every field in order, returning a kwargs dict.

    Imports `questionary` lazily so this module can still be imported
    (and its field specs/validators unit-tested) in headless
    environments without a real TTY.

    Returns `None` if the user cancels partway through (Ctrl+C / Esc),
    so the caller can bail out to the menu instead of running with
    incomplete input.
    """
    import questionary

    values: dict[str, Any] = {}
    for field in fields:
        default_str = "" if field.default is None else str(field.default)
        answer = questionary.text(
            f"{field.label}:",
            default=default_str,
            validate=_VALIDATORS[field.kind],
        ).ask()
        if answer is None:  # user pressed Ctrl+C / Esc
            return None
        values[field.name] = _CASTERS[field.kind](answer)
    return values


# ----------------------------------------------------------------------
# Field specs per nonlinear method (§6.2 "Required inputs")
# ----------------------------------------------------------------------

BISECTION_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**3 - x - 2"),
    Field("a", "a", "float", 1.0),
    Field("b", "b", "float", 2.0),
    Field("tol", "tol", "float", 1e-6),
    Field("max_iter", "max_iter", "int", 100),
]

REGULA_FALSI_FIELDS: list[Field] = BISECTION_FIELDS  # identical required inputs (§6.2)

FIXED_POINT_FIELDS: list[Field] = [
    Field("g", "g(x) = (rearranged so x = g(x))", "text", "(x + 2)**(1/3)"),
    Field("x0", "x0", "float", 1.5),
    Field("tol", "tol", "float", 1e-6),
    Field("max_iter", "max_iter", "int", 100),
]

NEWTON_RAPHSON_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**3 - x - 2"),
    Field("x0", "x0", "float", 1.5),
    Field("tol", "tol", "float", 1e-6),
    Field("max_iter", "max_iter", "int", 100),
]

SECANT_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**3 - x - 2"),
    Field("x0", "x0", "float", 1.0),
    Field("x1", "x1", "float", 2.0),
    Field("tol", "tol", "float", 1e-6),
    Field("max_iter", "max_iter", "int", 100),
]


# ----------------------------------------------------------------------
# Comparison mode (§7, Nonlinear Equations)
# ----------------------------------------------------------------------

def collect_comparison_inputs() -> Optional[dict[str, Any]]:
    """Collect one shared nonlinear "problem" for comparison mode.

    A single f(x) plus two points, reused as (a, b) for the bracket
    methods (Bisection/Regula Falsi) and as (x0, x1) for
    Newton-Raphson (x0 only) / Secant (x0 and x1) -- so "the same
    problem" (§7) means the same function and the same two numbers,
    interpreted per-method by the caller.
    """
    import questionary

    f_answer = questionary.text("f(x) =:", default="x**3 - x - 2", validate=_validate_text).ask()
    if f_answer is None:
        return None
    a_answer = questionary.text(
        "First point (bracket 'a' / starting point 'x0'):", default="1.0", validate=_validate_float
    ).ask()
    if a_answer is None:
        return None
    b_answer = questionary.text(
        "Second point (bracket 'b' / secant's 'x1'):", default="2.0", validate=_validate_float
    ).ask()
    if b_answer is None:
        return None
    tol_answer = questionary.text("tol:", default="1e-6", validate=_validate_float).ask()
    if tol_answer is None:
        return None
    max_iter_answer = questionary.text("max_iter:", default="100", validate=_validate_int).ask()
    if max_iter_answer is None:
        return None

    return {
        "f": f_answer,
        "a": float(a_answer),
        "b": float(b_answer),
        "tol": float(tol_answer),
        "max_iter": int(max_iter_answer),
    }


# ----------------------------------------------------------------------
# Field specs per numerical integration method (§6.5 "Required inputs")
# ----------------------------------------------------------------------

RECTANGLE_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2"),
    Field("a", "a", "float", 0.0),
    Field("b", "b", "float", 1.0),
    Field("n", "n", "int", 10),
    Field("side", "side (left/right)", "text", "left"),
]

MIDPOINT_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2"),
    Field("a", "a", "float", 0.0),
    Field("b", "b", "float", 1.0),
    Field("n", "n", "int", 10),
]

TRAPEZOIDAL_FIELDS: list[Field] = MIDPOINT_FIELDS  # identical required inputs (§6.5)

SIMPSON_1_3_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2"),
    Field("a", "a", "float", 0.0),
    Field("b", "b", "float", 1.0),
    Field("n", "n (must be even)", "int", 10),
]

SIMPSON_3_8_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2"),
    Field("a", "a", "float", 0.0),
    Field("b", "b", "float", 1.0),
    Field("n", "n (must be divisible by 3)", "int", 9),
]


# ----------------------------------------------------------------------
# Matrix/vector collection (§6.3, Linear Systems)
# ----------------------------------------------------------------------
# Linear systems methods take a matrix + vector(s) rather than a flat
# list of scalar fields, so they get their own collectors instead of
# going through `Field`/`collect_inputs` above.

def _prompt_dimension(label: str = "System size (n)", default: int = 3) -> Optional[int]:
    import questionary

    answer = questionary.text(f"{label}:", default=str(default), validate=_validate_int).ask()
    if answer is None:
        return None
    return int(answer)


def _prompt_row(label: str, n: int) -> Optional[list[float]]:
    import questionary

    def _validate_row(text: str) -> "bool | str":
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
        if len(parts) != n:
            return f"enter exactly {n} comma-separated number(s), got {len(parts)}"
        for p in parts:
            try:
                float(p)
            except ValueError:
                return f"'{p}' is not a valid number"
        return True

    answer = questionary.text(f"{label} ({n} comma-separated values):", validate=_validate_row).ask()
    if answer is None:
        return None
    return [float(p.strip()) for p in answer.split(",") if p.strip() != ""]


def collect_matrix(n: int, label: str = "A") -> Optional[list[list[float]]]:
    """Prompt for an n x n matrix, one row at a time."""
    rows: list[list[float]] = []
    for i in range(n):
        row = _prompt_row(f"{label} row {i + 1}", n)
        if row is None:
            return None
        rows.append(row)
    return rows


def collect_vector(n: int, label: str = "b") -> Optional[list[float]]:
    """Prompt for a single length-n vector."""
    return _prompt_row(label, n)


def collect_direct_system_inputs() -> Optional[dict[str, Any]]:
    """Collect A, b for the direct solvers: Gaussian elimination, Gauss-Jordan, LU."""
    n = _prompt_dimension()
    if n is None:
        return None
    A = collect_matrix(n, "A")
    if A is None:
        return None
    b = collect_vector(n, "b")
    if b is None:
        return None
    return {"A": A, "b": b}


def collect_matrix_only_inputs() -> Optional[dict[str, Any]]:
    """Collect A only, for matrix inverse."""
    n = _prompt_dimension("Matrix size (n)")
    if n is None:
        return None
    A = collect_matrix(n, "A")
    if A is None:
        return None
    return {"A": A}


def collect_iterative_system_inputs() -> Optional[dict[str, Any]]:
    """Collect A, b, x0, tol, max_iter for Jacobi / Gauss-Seidel."""
    import questionary

    n = _prompt_dimension()
    if n is None:
        return None
    A = collect_matrix(n, "A")
    if A is None:
        return None
    b = collect_vector(n, "b")
    if b is None:
        return None
    x0 = collect_vector(n, "x0 (initial guess)")
    if x0 is None:
        return None

    tol_answer = questionary.text("tol:", default="1e-6", validate=_validate_float).ask()
    if tol_answer is None:
        return None
    max_iter_answer = questionary.text("max_iter:", default="100", validate=_validate_int).ask()
    if max_iter_answer is None:
        return None

    return {"A": A, "b": b, "x0": x0, "tol": float(tol_answer), "max_iter": int(max_iter_answer)}


# ----------------------------------------------------------------------
# Point collection (§6.4, Interpolation)
# ----------------------------------------------------------------------
# Interpolation methods take a set of (x, y) data points plus a
# target x, rather than the linear-systems' matrix/vector shape or
# the nonlinear category's flat scalar fields, so they get their own
# collectors too.

def collect_points(n: int) -> Optional[tuple[list[float], list[float]]]:
    """Prompt for n (x, y) data points, one x and one y per point."""
    import questionary

    xs: list[float] = []
    ys: list[float] = []
    for i in range(n):
        x_answer = questionary.text(f"x_{i + 1}:", validate=_validate_float).ask()
        if x_answer is None:
            return None
        y_answer = questionary.text(f"y_{i + 1} = f(x_{i + 1}):", validate=_validate_float).ask()
        if y_answer is None:
            return None
        xs.append(float(x_answer))
        ys.append(float(y_answer))
    return xs, ys


def collect_fixed_interpolation_inputs(n: int) -> Optional[dict[str, Any]]:
    """Collect xs, ys (fixed count n) and x_target -- for linear/quadratic/cubic."""
    import questionary

    points = collect_points(n)
    if points is None:
        return None
    xs, ys = points

    x_target_answer = questionary.text("x_target:", validate=_validate_float).ask()
    if x_target_answer is None:
        return None
    return {"xs": xs, "ys": ys, "x_target": float(x_target_answer)}


def collect_linear_inputs() -> Optional[dict[str, Any]]:
    return collect_fixed_interpolation_inputs(2)


def collect_quadratic_inputs() -> Optional[dict[str, Any]]:
    return collect_fixed_interpolation_inputs(3)


def collect_cubic_inputs() -> Optional[dict[str, Any]]:
    return collect_fixed_interpolation_inputs(4)


def collect_variable_interpolation_inputs() -> Optional[dict[str, Any]]:
    """Collect n, xs, ys, and x_target -- for Lagrange / Newton divided-difference / Newton-Gregory."""
    n = _prompt_dimension("Number of points (n)", default=4)
    if n is None:
        return None
    return collect_fixed_interpolation_inputs(n)