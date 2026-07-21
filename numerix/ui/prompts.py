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