"""
ui/menu.py — category -> method -> input navigation (§7), via
questionary arrow-key select menus.

"Nonlinear Equations" and "Linear Systems" are fully wired to real
methods. Interpolation and Numerical Integration still show a "not
implemented yet" message if chosen (Phases 7-8) rather than crashing
— consistent with §8's "no unhandled exceptions reach the user under
any input".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from rich.console import Console

from numerix.core.interpolation import (
    cubic,
    lagrange,
    linear,
    newton_divided_diff,
    newton_gregory_backward,
    newton_gregory_forward,
    quadratic,
)
from numerix.core.linear_systems import (
    gauss_elimination,
    gauss_jordan,
    gauss_seidel,
    jacobi,
    lu_decomposition,
    matrix_inverse,
)
from numerix.core.nonlinear import bisection, fixed_point, newton_raphson, regula_falsi, secant
from numerix.ui import prompts
from numerix.ui.display import render_error, render_result

console = Console()

_BACK = "\u00ab Back"
_EXIT = "Exit"


@dataclass(frozen=True)
class MethodEntry:
    label: str
    collect: Callable[[], Optional[dict[str, Any]]]
    run: Callable[..., object]


def _scalar_collector(fields: list[prompts.Field]) -> Callable[[], Optional[dict[str, Any]]]:
    """Wrap a `Field` list for the generic scalar-input collection path (nonlinear methods)."""
    return lambda: prompts.collect_inputs(fields)


_NONLINEAR_METHODS: list[MethodEntry] = [
    MethodEntry("Bisection", _scalar_collector(prompts.BISECTION_FIELDS), bisection),
    MethodEntry("Regula Falsi", _scalar_collector(prompts.REGULA_FALSI_FIELDS), regula_falsi),
    MethodEntry("Fixed Point", _scalar_collector(prompts.FIXED_POINT_FIELDS), fixed_point),
    MethodEntry("Newton-Raphson", _scalar_collector(prompts.NEWTON_RAPHSON_FIELDS), newton_raphson),
    MethodEntry("Secant", _scalar_collector(prompts.SECANT_FIELDS), secant),
]

_LINEAR_SYSTEMS_METHODS: list[MethodEntry] = [
    MethodEntry("Gaussian Elimination", prompts.collect_direct_system_inputs, gauss_elimination),
    MethodEntry("Gauss-Jordan", prompts.collect_direct_system_inputs, gauss_jordan),
    MethodEntry("Matrix Inverse", prompts.collect_matrix_only_inputs, matrix_inverse),
    MethodEntry("LU Decomposition", prompts.collect_direct_system_inputs, lu_decomposition),
    MethodEntry("Jacobi", prompts.collect_iterative_system_inputs, jacobi),
    MethodEntry("Gauss-Seidel", prompts.collect_iterative_system_inputs, gauss_seidel),
]

_INTERPOLATION_METHODS: list[MethodEntry] = [
    MethodEntry("Linear Interpolation", prompts.collect_linear_inputs, linear),
    MethodEntry("Quadratic Interpolation", prompts.collect_quadratic_inputs, quadratic),
    MethodEntry("Cubic Interpolation", prompts.collect_cubic_inputs, cubic),
    MethodEntry("Lagrange Interpolation", prompts.collect_variable_interpolation_inputs, lagrange),
    MethodEntry("Newton Divided Difference", prompts.collect_variable_interpolation_inputs, newton_divided_diff),
    MethodEntry("Newton-Gregory Forward", prompts.collect_variable_interpolation_inputs, newton_gregory_forward),
    MethodEntry("Newton-Gregory Backward", prompts.collect_variable_interpolation_inputs, newton_gregory_backward),
]

_CATEGORIES: dict[str, list[MethodEntry]] = {
    "Nonlinear Equations": _NONLINEAR_METHODS,
    "Linear Systems": _LINEAR_SYSTEMS_METHODS,
    "Interpolation": _INTERPOLATION_METHODS,
    "Numerical Integration": [],
}


def run_menu() -> None:
    """Top-level menu loop: category -> method -> input -> result -> loop.

    Imports `questionary` lazily so this module (and its navigation
    logic) can still be imported and exercised without a real TTY.
    """
    import questionary

    while True:
        category = questionary.select(
            "Numerix — choose a category:",
            choices=list(_CATEGORIES.keys()) + [_EXIT],
        ).ask()

        if category is None or category == _EXIT:
            console.print("[dim]Goodbye.[/dim]")
            return

        _run_category(category)


def _run_category(category: str) -> None:
    import questionary

    methods = _CATEGORIES[category]

    if not methods:
        console.print(f"\n[yellow]{category} isn't implemented yet \u2014 coming in a later phase.[/yellow]\n")
        return

    while True:
        choice = questionary.select(
            f"{category} \u2014 choose a method:",
            choices=[m.label for m in methods] + [_BACK],
        ).ask()

        if choice is None or choice == _BACK:
            return

        entry = next(m for m in methods if m.label == choice)
        _run_method(entry)


def _run_method(entry: MethodEntry) -> None:
    import questionary

    values = entry.collect()
    if values is None:
        return  # user cancelled input part-way through

    try:
        result = entry.run(**values)
    except ValueError as exc:
        render_error(str(exc), console)
        return
    except Exception as exc:  # noqa: BLE001 -- last-resort safety net, per §8
        render_error(f"unexpected error: {exc}", console)
        return

    render_result(result, console)
    questionary.text("Press Enter to continue...").ask()