"""
ui/menu.py — category -> method -> input navigation (§7), via
questionary arrow-key select menus.

All four categories are fully wired to real methods. The Nonlinear
Equations category additionally offers a "Compare Methods" entry
(§7 comparison mode), and every single-method run offers to export
its `MethodResult` to `results/` as CSV or JSON (§7 export).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from rich.console import Console
from rich.table import Table

from numerix.core.integration import (
    midpoint_rule,
    rectangle_rule,
    simpson_1_3,
    simpson_3_8,
    trapezoidal_rule,
)
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
_COMPARE = "\u2696 Compare Methods"
_RESULTS_DIR = "results"


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

_INTEGRATION_METHODS: list[MethodEntry] = [
    MethodEntry("Rectangle Rule", _scalar_collector(prompts.RECTANGLE_FIELDS), rectangle_rule),
    MethodEntry("Midpoint Rule", _scalar_collector(prompts.MIDPOINT_FIELDS), midpoint_rule),
    MethodEntry("Trapezoidal Rule", _scalar_collector(prompts.TRAPEZOIDAL_FIELDS), trapezoidal_rule),
    MethodEntry("Simpson's 1/3 Rule", _scalar_collector(prompts.SIMPSON_1_3_FIELDS), simpson_1_3),
    MethodEntry("Simpson's 3/8 Rule", _scalar_collector(prompts.SIMPSON_3_8_FIELDS), simpson_3_8),
]

_CATEGORIES: dict[str, list[MethodEntry]] = {
    "Nonlinear Equations": _NONLINEAR_METHODS,
    "Linear Systems": _LINEAR_SYSTEMS_METHODS,
    "Interpolation": _INTERPOLATION_METHODS,
    "Numerical Integration": _INTEGRATION_METHODS,
}

# Comparison mode (§7): pick 2+ methods from the same category, run on
# the same problem, show a summary table side by side. Scoped to the
# nonlinear category per Phase 9 / §7 ("applies most usefully to the
# nonlinear category"). "Fixed Point" is deliberately excluded: it
# takes a rearranged g(x), not the f(x)/bracket shape shared by the
# other four, so it can't run on "the same problem" as they can.
_COMPARISON_CANDIDATES: dict[str, Callable[..., object]] = {
    "Bisection": bisection,
    "Regula Falsi": regula_falsi,
    "Newton-Raphson": newton_raphson,
    "Secant": secant,
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
        choices = [m.label for m in methods]
        if category == "Nonlinear Equations":
            choices.append(_COMPARE)
        choices.append(_BACK)

        choice = questionary.select(f"{category} \u2014 choose a method:", choices=choices).ask()

        if choice is None or choice == _BACK:
            return
        if choice == _COMPARE:
            _run_nonlinear_comparison()
            continue

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
    _offer_export(result)
    questionary.text("Press Enter to continue...").ask()


# ----------------------------------------------------------------------
# Comparison mode (§7)
# ----------------------------------------------------------------------

def _run_nonlinear_comparison() -> None:
    """Pick 2+ nonlinear methods, run them on the same problem, show a
    summary table side by side (§7).
    """
    import questionary

    values = prompts.collect_comparison_inputs()
    if values is None:
        return  # user cancelled part-way through

    selected = questionary.checkbox(
        "Select 2+ methods to compare (space to toggle, enter to confirm):",
        choices=list(_COMPARISON_CANDIDATES.keys()),
    ).ask()

    if not selected:
        return  # user cancelled (Ctrl+C) or confirmed with nothing selected
    if len(selected) < 2:
        render_error("pick at least 2 methods to compare.", console)
        return

    f, a, b, tol, max_iter = values["f"], values["a"], values["b"], values["tol"], values["max_iter"]
    results: dict[str, object] = {}
    errors: dict[str, str] = {}

    for label in selected:
        try:
            if label in ("Bisection", "Regula Falsi"):
                results[label] = _COMPARISON_CANDIDATES[label](f, a, b, tol=tol, max_iter=max_iter)
            elif label == "Newton-Raphson":
                results[label] = newton_raphson(f, a, tol=tol, max_iter=max_iter)
            else:  # Secant
                results[label] = secant(f, a, b, tol=tol, max_iter=max_iter)
        except ValueError as exc:
            errors[label] = str(exc)
        except Exception as exc:  # noqa: BLE001 -- last-resort safety net, per §8
            errors[label] = f"unexpected error: {exc}"

    _render_comparison_table(selected, results, errors)

    if results:
        _offer_export_many(results)
    questionary.text("Press Enter to continue...").ask()


def _render_comparison_table(order: list[str], results: dict, errors: dict) -> None:
    """Summary table: method, iterations, final error, converged? (§7)."""
    table = Table(title="Comparison", header_style="bold magenta")
    table.add_column("Method")
    table.add_column("Iterations", justify="right")
    table.add_column("Final Error", justify="right")
    table.add_column("Converged?")

    for label in order:
        if label in errors:
            table.add_row(label, "\u2014", "\u2014", f"[red]\u2717 {errors[label]}[/red]")
            continue
        result = results[label]
        converged_cell = "[green]\u2713[/green]" if result.converged else "[red]\u2717[/red]"
        error_cell = f"{result.approx_error:.6g}" if result.approx_error is not None else "\u2014"
        table.add_row(label, str(result.n_iterations), error_cell, converged_cell)

    console.print(table)


# ----------------------------------------------------------------------
# Export (§7): offer to save any MethodResult to results/ as CSV/JSON
# ----------------------------------------------------------------------

def _offer_export(result) -> None:
    import questionary

    choice = questionary.select("Export this result?", choices=["Skip", "Save as CSV", "Save as JSON"]).ask()
    if choice in (None, "Skip"):
        return
    fmt = "csv" if choice == "Save as CSV" else "json"
    try:
        path = result.save(_RESULTS_DIR, fmt)
        console.print(f"[green]Saved to {path}[/green]")
    except Exception as exc:  # noqa: BLE001 -- exporting must never crash the app
        render_error(f"could not save export: {exc}", console)


def _offer_export_many(results: dict) -> None:
    import questionary

    choice = questionary.select(
        "Export these results?", choices=["Skip", "Save all as CSV", "Save all as JSON"]
    ).ask()
    if choice in (None, "Skip"):
        return
    fmt = "csv" if choice == "Save all as CSV" else "json"
    for result in results.values():
        try:
            path = result.save(_RESULTS_DIR, fmt)
            console.print(f"[green]Saved to {path}[/green]")
        except Exception as exc:  # noqa: BLE001
            render_error(f"could not save export: {exc}", console)