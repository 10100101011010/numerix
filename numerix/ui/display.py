"""
ui/display.py — the single shared result renderer (§7).

Every algorithm returns a `MethodResult` (§5); this module is the
*only* place that knows how to print one. Every method, in every
category, renders through `render_result()` — no method invents its
own printing style.

Render order, per §7:
  banner (method name) -> input recap panel -> iteration table
  -> result panel -> stats footer -> converged/warning indicator
  (✓ green / ⚠ yellow / ✗ red)

The stats footer is method-aware (design/usability pass, pre-Phase
11): methods with a genuine convergence loop (bounded by tol/
max_iter) keep the "Iterations" label and show an "Approx. Error"
column; direct/tabular methods (linear solvers built as an
elimination/decomposition table, interpolation, integration formulas)
have no natural approximate-error concept, so that column is omitted
entirely rather than showing a dash, and the count column is
relabeled "Data Points" (interpolation) or "Table Size" (everything
else direct/tabular).
"""

from __future__ import annotations

from typing import Any

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from numerix.utils.result import MethodResult

# Neutral placeholder for a missing/blank iteration-table cell (e.g. a
# higher Newton-Gregory order that a given row doesn't reach).
_BLANK_CELL = "\u00b7"

# Methods with a genuine convergence loop (bounded by tol/max_iter),
# as opposed to a direct/tabular computation. Nonlinear solvers are
# always iterative; within Linear Systems, only Jacobi/Gauss-Seidel
# are (the rest -- Gaussian elimination, Gauss-Jordan, matrix
# inverse, LU -- are direct, just organized as a step-by-step table).
_ITERATIVE_LINEAR_METHODS = {"Jacobi", "Gauss-Seidel"}


def render_result(result: MethodResult, console: Console | None = None) -> None:
    """Render any `MethodResult` in the one consistent Numerix format.

    This is the single entry point the rest of the app should call —
    `ui/menu.py`, comparison mode, etc. (added in later phases) all
    funnel through this one function rather than printing themselves.
    """
    console = console or Console()

    _render_banner(result, console)
    _render_inputs(result, console)
    _render_iterations(result, console)
    _render_solution(result, console)
    _render_stats(result, console)
    _render_status(result, console)


def render_error(message: str, console: Console | None = None) -> None:
    """Render a friendly one-line error using the same red styling as a
    non-converged result (§7/§8). Never let a raw traceback reach the
    user — anything that can raise (bad domain, singular matrix,
    malformed function, divide-by-zero, ...) should be caught by the
    caller and passed through here instead.
    """
    console = console or Console()
    console.print(Panel(f"[bold red]✗ Error[/bold red]\n{message}", border_style="red", expand=False))


# ----------------------------------------------------------------------
# Section renderers
# ----------------------------------------------------------------------

def _render_banner(result: MethodResult, console: Console) -> None:
    banner = Panel(
        Align.center(f"[bold cyan]{result.method_name}[/bold cyan]"),
        subtitle=f"[dim]{result.category}[/dim]",
        border_style="cyan",
    )
    console.print(banner)


def _render_inputs(result: MethodResult, console: Console) -> None:
    if result.inputs:
        lines = "\n".join(f"[bold]{key}[/bold] = {_format_value(value)}" for key, value in result.inputs.items())
    else:
        lines = "[dim](no inputs)[/dim]"
    console.print(Panel(lines, title="Inputs", border_style="blue"))


def _render_iterations(result: MethodResult, console: Console) -> None:
    if not result.iterations:
        console.print(Panel("[dim]No iterations — exact solution found immediately.[/dim]", border_style="dim"))
        return

    # Column set = union of keys across all rows, in first-seen order,
    # since not every row is guaranteed to carry identical keys.
    columns: list[str] = []
    for row in result.iterations:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    table = Table(title="Iterations", header_style="bold magenta", show_lines=False)
    for col in columns:
        table.add_column(col, justify="right")
    for row in result.iterations:
        table.add_row(*[_format_cell(row.get(col)) for col in columns])

    console.print(table)


def _render_solution(result: MethodResult, console: Console) -> None:
    console.print(Panel(f"[bold]{_format_value(result.solution)}[/bold]", title="Result", border_style="white"))


def _render_stats(result: MethodResult, console: Console) -> None:
    iterative = _is_iterative(result)

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column(_count_label(result), justify="right")
    if iterative:
        table.add_column("Approx. Error", justify="right")
    table.add_column("Exec Time", justify="right")

    row = [str(result.n_iterations)]
    if iterative:
        row.append(_format_number(result.approx_error) if result.approx_error is not None else "—")
    row.append(f"{result.exec_time_ms:.3f} ms")

    table.add_row(*row)
    console.print(table)


def _render_status(result: MethodResult, console: Console) -> None:
    if result.warning:
        icon, color, label, message = "⚠", "yellow", "Warning", result.warning
    elif result.converged:
        icon, color, label, message = "✓", "green", "Converged", "Converged successfully."
    else:
        icon, color, label, message = "✗", "red", "Not Converged", (
            f"Did not meet tolerance within {result.n_iterations} iteration(s)."
        )

    console.print(Panel(f"[bold {color}]{icon} {label}[/bold {color}]\n{message}", border_style=color, expand=False))


# ----------------------------------------------------------------------
# Method-awareness helpers for the stats footer
# ----------------------------------------------------------------------

def _is_iterative(result: MethodResult) -> bool:
    """True for methods with a genuine tol/max_iter convergence loop."""
    if result.category == "Nonlinear Equations":
        return True
    if result.category == "Linear Systems":
        return result.method_name in _ITERATIVE_LINEAR_METHODS
    return False


def _count_label(result: MethodResult) -> str:
    """Label for the stats footer's leading count column."""
    if _is_iterative(result):
        return "Iterations"
    if result.category == "Interpolation":
        return "Data Points"
    return "Table Size"


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------

def _format_number(value: float) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_cell(value: Any) -> str:
    if value is None:
        return _BLANK_CELL
    if isinstance(value, str) and value.strip() == "":
        return _BLANK_CELL
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_value(value: Any) -> str:
    """Format an arbitrary `inputs`/`solution` value (float, list, or dict)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "\n".join(f"{k} = {_format_value(v)}" for k, v in value.items())
    return str(value)