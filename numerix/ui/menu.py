"""
ui/menu.py — category -> method -> input navigation (§7), via
questionary arrow-key select menus.

Only the "Nonlinear Equations" category is wired to real methods in
this phase. The other three categories already appear in the menu
(so the shape of the app doesn't change out from under Phases 6-8),
but selecting one just shows a "not implemented yet" message rather
than crashing — consistent with §8's "no unhandled exceptions reach
the user under any input".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.console import Console

from numerix.core.nonlinear import bisection, fixed_point, newton_raphson, regula_falsi, secant
from numerix.ui import prompts
from numerix.ui.display import render_error, render_result

console = Console()

_BACK = "\u00ab Back"
_EXIT = "Exit"


@dataclass(frozen=True)
class MethodEntry:
    label: str
    fields: list[prompts.Field]
    run: Callable[..., object]


_NONLINEAR_METHODS: list[MethodEntry] = [
    MethodEntry("Bisection", prompts.BISECTION_FIELDS, bisection),
    MethodEntry("Regula Falsi", prompts.REGULA_FALSI_FIELDS, regula_falsi),
    MethodEntry("Fixed Point", prompts.FIXED_POINT_FIELDS, fixed_point),
    MethodEntry("Newton-Raphson", prompts.NEWTON_RAPHSON_FIELDS, newton_raphson),
    MethodEntry("Secant", prompts.SECANT_FIELDS, secant),
]

_CATEGORIES: dict[str, list[MethodEntry]] = {
    "Nonlinear Equations": _NONLINEAR_METHODS,
    "Linear Systems": [],
    "Interpolation": [],
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

    values = prompts.collect_inputs(entry.fields)
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