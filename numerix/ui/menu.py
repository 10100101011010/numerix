"""
ui/menu.py — category -> method -> input navigation (§7), via
questionary arrow-key select menus.

All four categories are fully wired to real methods. Every category
now offers a "Compare Methods" entry (§7 comparison mode) alongside
its method list: pick 2+ methods from that category, run them on the
same shared problem, see a summary table side by side. Each category
defines its own "same problem" shape and its own candidate list
(excluding methods whose required inputs or solution shape don't
match the rest — documented next to each `_COMPARISON_*_CANDIDATES`
dict below).

The top-level menu holds the four categories plus a Help/About entry,
a single Settings entry (currently just the plot toggle, pulled out
of the category list), and Exit. Every single-method run offers to
export its `MethodResult` to `results/` as CSV or JSON (§7 export),
and offers an optional plot (§7 plotting) unless disabled via the
`--no-plot` flag or the Settings toggle.

Phase 11 polish pass (§7/§8 "no raw traceback ever reaches the
user"): every core-method call already raises only `ValueError` for
bad domains/singular matrices/malformed functions/invalid brackets,
and each of the five call sites that invoke one (`_run_method`, the
four `_run_*_comparison` functions) already catches `ValueError`
specifically for a clean message, then `Exception` generally as its
own last-resort net. `_run_safely` below is the *outer* layer on top
of that: it wraps every dispatch out of the menu loop itself --
input collection, comparison-mode input collection, rendering,
export, plotting, and the category/settings/help dispatch — so a bug
anywhere in that surrounding UI code (not just inside a core-method
call) still degrades to one friendly line instead of a crash.
`EOFError`/`KeyboardInterrupt` are deliberately re-raised through it
so a closed/interrupted session still exits cleanly via `main.py`
rather than looping on repeated error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
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
from numerix.ui import plots, prompts
from numerix.ui.display import render_error, render_result

console = Console()

_BACK = "\u00ab Back"
_EXIT = "Exit"
_COMPARE = "\u2696 Compare Methods"
_SETTINGS = "\u2699 Settings"
_HELP = "? Help / About"
_RESULTS_DIR = "results"


class _Settings:
    """Small mutable session state (currently just the plot toggle),
    threaded through the menu loop rather than made global so tests
    can drive `run_menu()` repeatedly without leaking state.
    """

    def __init__(self, plot_enabled: bool) -> None:
        self.plot_enabled = plot_enabled


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
# the same problem, show a summary table side by side. Every category
# below defines its own candidate list, excluding any method whose
# required inputs or solution shape doesn't match the rest of its
# category closely enough to count as "the same problem".

_COMPARISON_NONLINEAR_CANDIDATES: dict[str, Callable[..., object]] = {
    "Bisection": bisection,
    "Regula Falsi": regula_falsi,
    "Newton-Raphson": newton_raphson,
    "Secant": secant,
}
# "Fixed Point" is deliberately excluded: it takes a rearranged g(x),
# not the f(x)/bracket shape shared by the other four, so it can't
# run on "the same problem" as they can.

_COMPARISON_LINEAR_CANDIDATES: dict[str, Callable[..., object]] = {
    "Gaussian Elimination": gauss_elimination,
    "Gauss-Jordan": gauss_jordan,
    "Jacobi": jacobi,
    "Gauss-Seidel": gauss_seidel,
}
# "Matrix Inverse" is excluded: it takes A only (no b), so it isn't
# solving the same Ax = b problem as the other four. "LU Decomposition"
# is excluded too: it returns a dict solution ({"x", "L", "U"}) rather
# than a plain solution vector, which would break an apples-to-apples
# comparison against the vector the other four return.

_COMPARISON_INTERPOLATION_CANDIDATES: dict[str, Callable[..., object]] = {
    "Lagrange Interpolation": lagrange,
    "Newton Divided Difference": newton_divided_diff,
    "Newton-Gregory Forward": newton_gregory_forward,
    "Newton-Gregory Backward": newton_gregory_backward,
}
# "Linear", "Quadratic", and "Cubic" Interpolation are excluded: each
# requires an exact fixed point count (2/3/4 respectively) rather
# than the shared variable-n point set the other four accept, so they
# can't run on "the same problem" as one another or as this group.

_COMPARISON_INTEGRATION_CANDIDATES: dict[str, Callable[..., object]] = {
    "Rectangle Rule": rectangle_rule,
    "Midpoint Rule": midpoint_rule,
    "Trapezoidal Rule": trapezoidal_rule,
    "Simpson's 1/3 Rule": simpson_1_3,
    "Simpson's 3/8 Rule": simpson_3_8,
}
# All five share the same f/a/b/n shape, so no exclusions here.
# Rectangle Rule's extra "side" parameter is fixed to "left" in
# comparison mode (see `_run_integration_comparison`) rather than
# asked separately, since it's an implementation detail of that one
# rule, not part of the shared problem the other four take unmodified.


def _run_safely(fn: Callable[..., None], *args: Any) -> None:
    """Last-resort safety net (§7/§8): run `fn`, and if anything escapes
    it beyond the ValueError/Exception handling `fn` already does
    around its own core-method call, catch it here and print one
    friendly line instead of letting a raw traceback reach the user.

    This is the *outer* layer, used at every dispatch point out of
    the menu loop (`_run_method`, each comparison runner, Settings,
    Help) -- so a bug in the surrounding UI code (input collection,
    rendering, export, plotting, the menu loop itself) is covered
    too, not just the core-method call each of those already guards
    internally.

    `EOFError`/`KeyboardInterrupt` are re-raised untouched so a
    closed/interrupted session still exits cleanly through the
    handler in `main.py`, instead of this net catching them and the
    loop immediately hitting the same closed input again.
    """
    try:
        fn(*args)
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001
        render_error(f"unexpected error: {exc}", console)


def run_menu(plot_enabled: bool = True) -> None:
    """Top-level menu loop: category -> method -> input -> result -> loop.

    `plot_enabled` mirrors the `--no-plot` CLI flag (§7): pass
    `plot_enabled=False` to suppress plot offers for the whole
    session. It can also be flipped mid-session from the Settings
    submenu.

    Imports `questionary` lazily so this module (and its navigation
    logic) can still be imported and exercised without a real TTY.
    """
    import questionary

    settings = _Settings(plot_enabled)

    while True:
        status = "ON" if settings.plot_enabled else "OFF"
        console.print(f"[dim](Plotting: {status})[/dim]")
        category = questionary.select(
            "Numerix — choose a category:",
            choices=list(_CATEGORIES.keys()) + [_HELP, _SETTINGS, _EXIT],
        ).ask()

        if category is None or category == _EXIT:
            console.print("[dim]Goodbye.[/dim]")
            return
        if category == _HELP:
            _run_safely(_run_help)
            continue
        if category == _SETTINGS:
            _run_safely(_run_settings, settings)
            continue

        _run_safely(_run_category, category, settings)


def _run_settings(settings: _Settings) -> None:
    """Settings submenu: currently just the plot toggle, pulled out of
    the top-level category list so that list holds only the four math
    categories plus Settings and Exit.
    """
    import questionary

    while True:
        status = "ON" if settings.plot_enabled else "OFF"
        choice = questionary.select(
            "Settings:",
            choices=[f"Toggle Plotting (currently {status})", _BACK],
        ).ask()

        if choice is None or choice == _BACK:
            return
        settings.plot_enabled = not settings.plot_enabled


def _run_help() -> None:
    """Help/About screen (§7 polish pass), reachable from the top-level
    menu. Method list and count are built from `_CATEGORIES` itself
    rather than hardcoded, so this can never drift out of sync with
    what's actually registered.
    """
    import questionary

    method_lines = "\n".join(
        f"  [bold]{category}[/bold]: {', '.join(m.label for m in methods)}"
        for category, methods in _CATEGORIES.items()
        if methods
    )
    total_methods = sum(len(methods) for methods in _CATEGORIES.values())

    body = (
        "[bold cyan]Numerix[/bold cyan] — classical numerical methods, in your terminal.\n\n"
        f"Implements {total_methods} methods across four categories, entirely from first "
        "principles (no library solvers — Gaussian elimination, Newton-Raphson, and the "
        "rest are all hand-written). Every result renders through one consistent format: "
        "input recap \u2192 iteration table \u2192 result \u2192 stats/converged footer.\n\n"
        "[bold]Methods by category[/bold]\n"
        f"{method_lines}\n\n"
        "[bold]Navigating[/bold]\n"
        "  Up/Down + Enter to choose. In Compare Methods, Space toggles a checkbox and "
        "Enter confirms the selection. Esc / Ctrl+C backs out of any prompt or submenu.\n\n"
        "[bold]Defaults[/bold]\n"
        "  tol = 1e-6 and max_iter = 100 on every iterative method (Bisection, Regula "
        "Falsi, Fixed Point, Newton-Raphson, Secant, Jacobi, Gauss-Seidel) — both are "
        "pre-filled on the prompt and can be overridden by typing a different value.\n\n"
        "[bold]Typing functions[/bold]\n"
        "  Plain math text, e.g. x**3 - x - 2, sin(x) - x/2. Allowed: sin, cos, tan, asin, "
        "acos, atan, sinh, cosh, tanh, exp, log, ln, sqrt, abs, floor, ceiling, pi, e.\n\n"
        "[bold]After a run[/bold]\n"
        "  Every result (single or comparison) can be exported to results/ as CSV or "
        "JSON, and — if plotting is on in Settings — shown as an optional matplotlib "
        "plot.\n\n"
        "[bold]Errors[/bold]\n"
        "  Invalid brackets, singular matrices, malformed functions, bad domains, and "
        "non-convergence all show one short message here, never a raw error trace."
    )

    console.print(Panel(body, title="Help / About", border_style="cyan"))
    questionary.text("Press Enter to return to the menu...").ask()


def _run_category(category: str, settings: _Settings) -> None:
    import questionary

    methods = _CATEGORIES[category]

    if not methods:
        console.print(f"\n[yellow]{category} isn't implemented yet \u2014 coming in a later phase.[/yellow]\n")
        return

    while True:
        choices = [m.label for m in methods] + [_COMPARE, _BACK]

        choice = questionary.select(f"{category} \u2014 choose a method:", choices=choices).ask()

        if choice is None or choice == _BACK:
            return
        if choice == _COMPARE:
            _run_safely(_COMPARISON_RUNNERS[category])
            continue

        entry = next(m for m in methods if m.label == choice)
        _run_safely(_run_method, entry, settings)


def _run_method(entry: MethodEntry, settings: _Settings) -> None:
    import questionary

    # entry.collect() and everything below it (render/export/plot) are
    # covered by the _run_safely() wrap around this whole call in
    # _run_category — only the core-method call right below gets its
    # own narrower ValueError handling, for a cleaner message.
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
    if settings.plot_enabled:
        _offer_plot(result)
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
        choices=list(_COMPARISON_NONLINEAR_CANDIDATES.keys()),
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
                results[label] = _COMPARISON_NONLINEAR_CANDIDATES[label](f, a, b, tol=tol, max_iter=max_iter)
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


def _run_linear_comparison() -> None:
    """Pick 2+ linear-systems methods, run them on the same A, b (+ x0,
    tol, max_iter for the iterative pair), show a summary table side
    by side (§7).
    """
    import questionary

    values = prompts.collect_linear_comparison_inputs()
    if values is None:
        return

    selected = questionary.checkbox(
        "Select 2+ methods to compare (space to toggle, enter to confirm):",
        choices=list(_COMPARISON_LINEAR_CANDIDATES.keys()),
    ).ask()

    if not selected:
        return
    if len(selected) < 2:
        render_error("pick at least 2 methods to compare.", console)
        return

    A, b, x0, tol, max_iter = values["A"], values["b"], values["x0"], values["tol"], values["max_iter"]
    results: dict[str, object] = {}
    errors: dict[str, str] = {}

    for label in selected:
        try:
            if label in ("Jacobi", "Gauss-Seidel"):
                results[label] = _COMPARISON_LINEAR_CANDIDATES[label](A, b, x0, tol=tol, max_iter=max_iter)
            else:  # Gaussian Elimination, Gauss-Jordan -- direct, ignore x0/tol/max_iter
                results[label] = _COMPARISON_LINEAR_CANDIDATES[label](A, b)
        except ValueError as exc:
            errors[label] = str(exc)
        except Exception as exc:  # noqa: BLE001
            errors[label] = f"unexpected error: {exc}"

    _render_comparison_table(selected, results, errors)

    if results:
        _offer_export_many(results)
    questionary.text("Press Enter to continue...").ask()


def _run_interpolation_comparison() -> None:
    """Pick 2+ interpolation methods, run them on the same point set and
    x_target, show a summary table side by side (§7).
    """
    import questionary

    values = prompts.collect_variable_interpolation_inputs()
    if values is None:
        return

    selected = questionary.checkbox(
        "Select 2+ methods to compare (space to toggle, enter to confirm):",
        choices=list(_COMPARISON_INTERPOLATION_CANDIDATES.keys()),
    ).ask()

    if not selected:
        return
    if len(selected) < 2:
        render_error("pick at least 2 methods to compare.", console)
        return

    results: dict[str, object] = {}
    errors: dict[str, str] = {}

    for label in selected:
        try:
            results[label] = _COMPARISON_INTERPOLATION_CANDIDATES[label](**values)
        except ValueError as exc:
            errors[label] = str(exc)
        except Exception as exc:  # noqa: BLE001
            errors[label] = f"unexpected error: {exc}"

    _render_comparison_table(selected, results, errors)

    if results:
        _offer_export_many(results)
    questionary.text("Press Enter to continue...").ask()


def _run_integration_comparison() -> None:
    """Pick 2+ integration rules, run them on the same f, a, b, n, show a
    summary table side by side (§7).
    """
    import questionary

    values = prompts.collect_integration_comparison_inputs()
    if values is None:
        return

    selected = questionary.checkbox(
        "Select 2+ methods to compare (space to toggle, enter to confirm):",
        choices=list(_COMPARISON_INTEGRATION_CANDIDATES.keys()),
    ).ask()

    if not selected:
        return
    if len(selected) < 2:
        render_error("pick at least 2 methods to compare.", console)
        return

    f, a, b, n = values["f"], values["a"], values["b"], values["n"]
    results: dict[str, object] = {}
    errors: dict[str, str] = {}

    for label in selected:
        try:
            if label == "Rectangle Rule":
                results[label] = rectangle_rule(f, a, b, n, side="left")
            else:
                results[label] = _COMPARISON_INTEGRATION_CANDIDATES[label](f, a, b, n)
        except ValueError as exc:
            errors[label] = str(exc)
        except Exception as exc:  # noqa: BLE001
            errors[label] = f"unexpected error: {exc}"

    _render_comparison_table(selected, results, errors)

    if results:
        _offer_export_many(results)
    questionary.text("Press Enter to continue...").ask()


_COMPARISON_RUNNERS: dict[str, Callable[[], None]] = {
    "Nonlinear Equations": _run_nonlinear_comparison,
    "Linear Systems": _run_linear_comparison,
    "Interpolation": _run_interpolation_comparison,
    "Numerical Integration": _run_integration_comparison,
}


def _format_comparison_result(value: Any) -> str:
    """Compact one-line form of a `MethodResult.solution` for the
    comparison table's Result column (float, vector, or fallback str).
    """
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in value) + "]"
    return str(value)


def _render_comparison_table(order: list[str], results: dict, errors: dict) -> None:
    """Summary table: method, result, iterations, final error, converged? (§7)."""
    table = Table(title="Comparison", header_style="bold magenta")
    table.add_column("Method")
    table.add_column("Result")
    table.add_column("Iterations", justify="right")
    table.add_column("Final Error", justify="right")
    table.add_column("Converged?")

    for label in order:
        if label in errors:
            table.add_row(label, "\u2014", "\u2014", "\u2014", f"[red]\u2717 {errors[label]}[/red]")
            continue
        result = results[label]
        converged_cell = "[green]\u2713[/green]" if result.converged else "[red]\u2717[/red]"
        error_cell = f"{result.approx_error:.6g}" if result.approx_error is not None else "\u2014"
        table.add_row(
            label,
            _format_comparison_result(result.solution),
            str(result.n_iterations),
            error_cell,
            converged_cell,
        )

    console.print(table)


def _offer_plot(result) -> None:
    import questionary

    choice = questionary.select("Show a plot of this result?", choices=["Skip", "Yes"]).ask()
    if choice == "Yes":
        plots.plot_result(result, console)


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