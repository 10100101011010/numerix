"""
main.py — minimal entry point (Phase 4).

This is intentionally bare: plain numbered `input()` prompts, no
arrow-key menu yet (that's `ui/menu.py` + `ui/prompts.py` in Phase 5).
Its only job right now is to prove the pipeline works end-to-end --
parse -> solve -> MethodResult -> render_result -- and that no bad
input produces a raw traceback (§8).
"""

from __future__ import annotations

from rich.console import Console

from numerix.core.nonlinear import bisection, fixed_point, newton_raphson, regula_falsi, secant
from numerix.ui.display import render_error, render_result

console = Console()

_METHODS = {
    "1": ("Bisection", bisection),
    "2": ("Regula Falsi", regula_falsi),
    "3": ("Fixed Point", fixed_point),
    "4": ("Newton-Raphson", newton_raphson),
    "5": ("Secant", secant),
}


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw if raw else (default or "")


def _prompt_float(label: str, default: float) -> float:
    raw = _prompt(label, str(default))
    try:
        return float(raw)
    except ValueError:
        console.print(f"[yellow]Couldn't read '{raw}' as a number, using default {default}.[/yellow]")
        return default


def _prompt_int(label: str, default: int) -> int:
    raw = _prompt(label, str(default))
    try:
        return int(raw)
    except ValueError:
        console.print(f"[yellow]Couldn't read '{raw}' as an integer, using default {default}.[/yellow]")
        return default


def _run_once() -> None:
    console.print("\n[bold]Numerix[/bold] — [dim]nonlinear equation solvers (demo)[/dim]\n")
    console.print("1) Bisection\n2) Regula Falsi\n3) Fixed Point\n4) Newton-Raphson\n5) Secant\n")

    choice = _prompt("Choose a method (1-5)", "4")
    if choice not in _METHODS:
        render_error(f"'{choice}' is not a valid choice (expected 1-5).", console)
        return

    name, method_fn = _METHODS[choice]
    tol = 1e-6
    max_iter = 100

    try:
        if name in ("Bisection", "Regula Falsi"):
            f = _prompt("f(x) =", "x**3 - x - 2")
            a = _prompt_float("a", 1.0)
            b = _prompt_float("b", 2.0)
            tol = _prompt_float("tol", tol)
            max_iter = _prompt_int("max_iter", max_iter)
            result = method_fn(f, a, b, tol=tol, max_iter=max_iter)

        elif name == "Fixed Point":
            g = _prompt("g(x) = (rearranged so x = g(x))", "(x + 2)**(1/3)")
            x0 = _prompt_float("x0", 1.5)
            tol = _prompt_float("tol", tol)
            max_iter = _prompt_int("max_iter", max_iter)
            result = fixed_point(g, x0, tol=tol, max_iter=max_iter)

        elif name == "Newton-Raphson":
            f = _prompt("f(x) =", "x**3 - x - 2")
            x0 = _prompt_float("x0", 1.5)
            tol = _prompt_float("tol", tol)
            max_iter = _prompt_int("max_iter", max_iter)
            result = newton_raphson(f, x0, tol=tol, max_iter=max_iter)

        else:  # Secant
            f = _prompt("f(x) =", "x**3 - x - 2")
            x0 = _prompt_float("x0", 1.0)
            x1 = _prompt_float("x1", 2.0)
            tol = _prompt_float("tol", tol)
            max_iter = _prompt_int("max_iter", max_iter)
            result = secant(f, x0, x1, tol=tol, max_iter=max_iter)

    except ValueError as exc:
        render_error(str(exc), console)
        return
    except Exception as exc:  # noqa: BLE001 -- last-resort safety net, per §8
        render_error(f"unexpected error: {exc}", console)
        return

    render_result(result, console)


def main() -> None:
    """Entry point (also wired as the `numerix` console script in pyproject.toml)."""
    try:
        while True:
            _run_once()
            again = input("\nRun another? (y/n) [y]: ").strip().lower()
            if again == "n":
                break
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()