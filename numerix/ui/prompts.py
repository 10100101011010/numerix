"""
ui/prompts.py — collects and validates user input per method (§7).

Each nonlinear method has a small, declarative list of `Field`s (in
the same order as Section 6.2's "Required inputs" column).
`collect_inputs` walks that list with `questionary.text` (or
`questionary.select` for choice-based fields), validating each answer
before moving on, and returns a plain dict ready to `**kwargs`
straight into the matching `core` function.

Every prompt shows a short inline example. That example is always a
real, working `default=` — pressing Enter accepts it as-is, typing
anything overrides it. Nothing in this file shows an example via
`instruction=` alone without also wiring it up as `default=`: a
hint that looks usable but isn't (submits empty text and fails
validation on Enter) is treated as a bug here, not a style choice.

Placeholder rule for *sequential same-kind* prompts (matrix rows,
interpolation points): the example must vary per position and, taken
literally as a full answer (i.e. accepted by pressing Enter through
every prompt), must produce valid, non-degenerate input -- distinct
x-values for interpolation points, a non-singular matrix for matrix
rows. A single fixed example repeated at every prompt (e.g. always
"2, -1, 0") would walk a user who just hits Enter repeatedly straight
into duplicate x-values or a singular matrix, so those placeholders
are generated per-index instead of hardcoded. See
`_matrix_row_placeholder` and the `collect_points` placeholders below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Field:
    name: str                                         # kwarg name expected by the core function
    label: str                                         # text shown to the user
    kind: str                                          # "text" | "float" | "int" | "select"
    default: Any = None
    placeholder: Optional[str] = None                  # short example shown as an inline instruction
    choices: Optional[list[tuple[str, str]]] = None    # (display label, value) pairs, kind="select" only


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


def _fmt(value: float) -> str:
    """Compact display form for a numeric placeholder (e.g. `4` not `4.0`, `-1` not `-1.0`)."""
    return f"{value:g}"


def collect_inputs(fields: list[Field]) -> Optional[dict[str, Any]]:
    """Prompt for every field in order, returning a kwargs dict.

    Text/float/int fields go through `questionary.text` (with a real
    `default=` drawn from `Field.default` and a short "(e.g. ...)"
    instruction drawn from `Field.placeholder`, so pressing Enter
    accepts the shown example); `select` fields go through
    `questionary.select` over `Field.choices`, with the chosen label
    mapped back to its underlying value before being added to the
    returned dict.

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
        if field.kind == "select":
            choices = field.choices or []
            labels = [label for label, _ in choices]
            default_label = next(
                (label for label, value in choices if value == field.default), labels[0] if labels else None
            )
            answer = questionary.select(f"{field.label}:", choices=labels, default=default_label).ask()
            if answer is None:  # user pressed Ctrl+C / Esc
                return None
            values[field.name] = next(value for label, value in choices if label == answer)
            continue

        default_str = "" if field.default is None else str(field.default)
        instruction = f"(e.g. {field.placeholder})" if field.placeholder else None
        answer = questionary.text(
            f"{field.label}:",
            default=default_str,
            validate=_VALIDATORS[field.kind],
            instruction=instruction,
        ).ask()
        if answer is None:  # user pressed Ctrl+C / Esc
            return None
        values[field.name] = _CASTERS[field.kind](answer)
    return values


# ----------------------------------------------------------------------
# Field specs per nonlinear method (§6.2 "Required inputs")
# ----------------------------------------------------------------------
# Every field here appears once per method (no looped/sequential
# same-kind prompts). Each already carries a real `default=` matching
# its `placeholder=` -- audited, no change needed.

BISECTION_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**3 - x - 2", placeholder="x**3 - x - 2"),
    Field("a", "a", "float", 1.0, placeholder="1.0"),
    Field("b", "b", "float", 2.0, placeholder="2.0"),
    Field("tol", "tol", "float", 1e-6, placeholder="1e-6"),
    Field("max_iter", "max_iter", "int", 100, placeholder="100"),
]

REGULA_FALSI_FIELDS: list[Field] = BISECTION_FIELDS  # identical required inputs (§6.2)

FIXED_POINT_FIELDS: list[Field] = [
    Field("g", "g(x) = (rearranged so x = g(x))", "text", "(x + 2)**(1/3)", placeholder="(x + 2)**(1/3)"),
    Field("x0", "x0", "float", 1.5, placeholder="1.5"),
    Field("tol", "tol", "float", 1e-6, placeholder="1e-6"),
    Field("max_iter", "max_iter", "int", 100, placeholder="100"),
]

NEWTON_RAPHSON_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**3 - x - 2", placeholder="x**3 - x - 2"),
    Field("x0", "x0", "float", 1.5, placeholder="1.5"),
    Field("tol", "tol", "float", 1e-6, placeholder="1e-6"),
    Field("max_iter", "max_iter", "int", 100, placeholder="100"),
]

SECANT_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**3 - x - 2", placeholder="x**3 - x - 2"),
    Field("x0", "x0", "float", 1.0, placeholder="1.0"),
    Field("x1", "x1", "float", 2.0, placeholder="2.0"),
    Field("tol", "tol", "float", 1e-6, placeholder="1e-6"),
    Field("max_iter", "max_iter", "int", 100, placeholder="100"),
]


# ----------------------------------------------------------------------
# Comparison mode (§7, Nonlinear Equations)
# ----------------------------------------------------------------------
# f / a / b / tol / max_iter, each asked once, each with a real
# default= already -- audited, no change needed.

def collect_comparison_inputs() -> Optional[dict[str, Any]]:
    """Collect one shared nonlinear "problem" for comparison mode.

    A single f(x) plus two points, reused as (a, b) for the bracket
    methods (Bisection/Regula Falsi) and as (x0, x1) for
    Newton-Raphson (x0 only) / Secant (x0 and x1) -- so "the same
    problem" (§7) means the same function and the same two numbers,
    interpreted per-method by the caller.
    """
    import questionary

    f_answer = questionary.text(
        "f(x) =:", default="x**3 - x - 2", validate=_validate_text, instruction="(e.g. x**3 - x - 2)"
    ).ask()
    if f_answer is None:
        return None
    a_answer = questionary.text(
        "First point (bracket 'a' / starting point 'x0'):",
        default="1.0",
        validate=_validate_float,
        instruction="(e.g. 1.0)",
    ).ask()
    if a_answer is None:
        return None
    b_answer = questionary.text(
        "Second point (bracket 'b' / secant's 'x1'):",
        default="2.0",
        validate=_validate_float,
        instruction="(e.g. 2.0)",
    ).ask()
    if b_answer is None:
        return None
    tol_answer = questionary.text(
        "tol:", default="1e-6", validate=_validate_float, instruction="(e.g. 1e-6)"
    ).ask()
    if tol_answer is None:
        return None
    max_iter_answer = questionary.text(
        "max_iter:", default="100", validate=_validate_int, instruction="(e.g. 100)"
    ).ask()
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
# f / a / b / n / side, each asked once, each with a real default=
# already -- audited, no change needed.

RECTANGLE_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2", placeholder="x**2"),
    Field("a", "a", "float", 0.0, placeholder="0.0"),
    Field("b", "b", "float", 1.0, placeholder="1.0"),
    Field("n", "n", "int", 10, placeholder="10"),
    Field(
        "side",
        "side",
        "select",
        "left",
        choices=[("Left Rectangle", "left"), ("Right Rectangle", "right")],
    ),
]

MIDPOINT_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2", placeholder="x**2"),
    Field("a", "a", "float", 0.0, placeholder="0.0"),
    Field("b", "b", "float", 1.0, placeholder="1.0"),
    Field("n", "n", "int", 10, placeholder="10"),
]

TRAPEZOIDAL_FIELDS: list[Field] = MIDPOINT_FIELDS  # identical required inputs (§6.5)

SIMPSON_1_3_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2", placeholder="x**2"),
    Field("a", "a", "float", 0.0, placeholder="0.0"),
    Field("b", "b", "float", 1.0, placeholder="1.0"),
    Field("n", "n (must be even)", "int", 10, placeholder="10, must be even"),
]

SIMPSON_3_8_FIELDS: list[Field] = [
    Field("f", "f(x) =", "text", "x**2", placeholder="x**2"),
    Field("a", "a", "float", 0.0, placeholder="0.0"),
    Field("b", "b", "float", 1.0, placeholder="1.0"),
    Field("n", "n (must be divisible by 3)", "int", 9, placeholder="9, must be divisible by 3"),
]


# ----------------------------------------------------------------------
# Matrix/vector collection (§6.3, Linear Systems)
# ----------------------------------------------------------------------
# Linear systems methods take a matrix + vector(s) rather than a flat
# list of scalar fields, so they get their own collectors instead of
# going through `Field`/`collect_inputs` above.

def _prompt_dimension(label: str = "System size (n)", default: int = 3) -> Optional[int]:
    import questionary

    answer = questionary.text(
        f"{label}:", default=str(default), validate=_validate_int, instruction=f"(e.g. {default})"
    ).ask()
    if answer is None:
        return None
    return int(answer)


def _matrix_row_placeholder(i: int, n: int) -> str:
    """Placeholder for matrix row `i` (0-indexed) of an n x n matrix.

    Generates row `i` of a small tridiagonal, diagonally dominant
    template: `4` on the diagonal, `-1` on the immediate neighbors,
    `0` elsewhere. Diagonal dominance guarantees the matrix is
    non-singular, so a user who accepts every row's default by
    pressing Enter ends up with a valid, solvable system -- not the
    same row repeated n times (which would be singular). Diagonal
    dominance is also exactly the property Jacobi/Gauss-Seidel need
    for guaranteed convergence, so the same template doubles as a
    good example there too.
    """
    cells = []
    for j in range(n):
        if j == i:
            cells.append("4")
        elif j == i - 1 or j == i + 1:
            cells.append("-1")
        else:
            cells.append("0")
    return ", ".join(cells)


def _vector_placeholder(n: int, value_at: Callable[[int], float]) -> str:
    """Placeholder for a length-n vector, one example value per position."""
    return ", ".join(_fmt(value_at(i)) for i in range(n))


def _prompt_row(label: str, n: int, placeholder: str) -> Optional[list[float]]:
    """Prompt for one comma-separated row/vector of length n.

    `placeholder` doubles as both the shown example (`instruction=`)
    and the real `default=` -- it's already validated CSV of the
    right length, so pressing Enter submits it as-is and passes
    `_validate_row` the same way typing it out by hand would.
    """
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

    answer = questionary.text(
        f"{label} ({n} comma-separated values):",
        default=placeholder,
        validate=_validate_row,
        instruction=f"(e.g. {placeholder})",
    ).ask()
    if answer is None:
        return None
    return [float(p.strip()) for p in answer.split(",") if p.strip() != ""]


def collect_matrix(n: int, label: str = "A") -> Optional[list[list[float]]]:
    """Prompt for an n x n matrix, one row at a time.

    Each row's placeholder comes from `_matrix_row_placeholder(i, n)`
    and is wired up as a real `default=`, so it differs per row and
    the full set is non-singular whether the user types it out or
    just presses Enter through every row -- fixes a bug where every
    row showed the identical example (e.g. "2, -1, 0" every time)
    with no working default, which would either produce a singular
    matrix (if typed literally) or fail validation outright (if
    accepted via Enter, since the hint wasn't a real default).
    """
    rows: list[list[float]] = []
    for i in range(n):
        row = _prompt_row(f"{label} row {i + 1}", n, placeholder=_matrix_row_placeholder(i, n))
        if row is None:
            return None
        rows.append(row)
    return rows


def collect_vector(n: int, label: str = "b", placeholder: Optional[str] = None) -> Optional[list[float]]:
    """Prompt for a single length-n vector.

    Defaults to a `1, 2, 3, ...` example sized to `n` (rather than a
    fixed 3-number string like "5, -3, 2" that silently stopped
    matching the required count once `n != 3`), wired up as a real
    `default=` so pressing Enter accepts it. Callers that want a
    different example (e.g. an all-zero initial guess) pass their own
    `placeholder`.
    """
    if placeholder is None:
        placeholder = _vector_placeholder(n, lambda i: i + 1)
    return _prompt_row(label, n, placeholder)


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
    x0 = collect_vector(n, "x0 (initial guess)", placeholder=_vector_placeholder(n, lambda i: 0.0))
    if x0 is None:
        return None

    tol_answer = questionary.text(
        "tol:", default="1e-6", validate=_validate_float, instruction="(e.g. 1e-6)"
    ).ask()
    if tol_answer is None:
        return None
    max_iter_answer = questionary.text(
        "max_iter:", default="100", validate=_validate_int, instruction="(e.g. 100)"
    ).ask()
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
    """Prompt for n (x, y) data points, one x and one y per point.

    Placeholders vary per point and are wired up as real defaults:
    x_i defaults to `i` (0, 1, 2, 3, ...) so a user who just presses
    Enter through every prompt ends up with n distinct, evenly-spaced
    x-values -- a valid, non-degenerate interpolation problem, and
    also exactly what Newton-Gregory forward/backward expect. y_i
    defaults to `i**2`, tracing out a simple quadratic so the example
    y's vary too rather than all matching. This fixes a bug where
    every point showed the same "x=1.0 / y=2.0" example with no
    working default -- accepting it via Enter past the first point
    would repeat x-values (a duplicate x breaks every interpolation
    method here).
    """
    import questionary

    xs: list[float] = []
    ys: list[float] = []
    for i in range(n):
        x_placeholder = _fmt(float(i))
        y_placeholder = _fmt(float(i * i))
        x_answer = questionary.text(
            f"x_{i + 1}:",
            default=x_placeholder,
            validate=_validate_float,
            instruction=f"(e.g. {x_placeholder})",
        ).ask()
        if x_answer is None:
            return None
        y_answer = questionary.text(
            f"y_{i + 1} = f(x_{i + 1}):",
            default=y_placeholder,
            validate=_validate_float,
            instruction=f"(e.g. {y_placeholder})",
        ).ask()
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

    x_target_answer = questionary.text(
        "x_target:", default="1.5", validate=_validate_float, instruction="(e.g. 1.5)"
    ).ask()
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