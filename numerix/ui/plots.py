"""
ui/plots.py — optional matplotlib plotting (§7).

One plot type per category that has a natural visual:
- Nonlinear Equations: f(x) curve + root marker
- Interpolation: data points + reconstructed interpolation curve
- Numerical Integration: f(x) curve + shaded area under the curve
(Linear Systems has no plot type defined in §7 -- `plot_result` skips
it gracefully rather than guessing.)

This is entirely optional and must never crash the app:
- If matplotlib isn't installed at all, catch the ImportError and
  print a "plot skipped" message.
- If matplotlib IS installed but there's no usable display (headless
  terminal, no DISPLAY, no GUI toolkit), matplotlib typically falls
  back to a non-interactive backend like "Agg" on its own -- and
  `plt.show()` on such a backend does not raise, it just silently
  does nothing. So `_show_or_save` checks the active backend first;
  if it's non-interactive (or `plt.show()` raises anyway), it saves a
  PNG to `plots/` instead and tells the user, rather than pretending
  to have shown something it didn't.

`matplotlib` is imported lazily inside each function (not at module
level) so this module -- and everything that imports it -- stays
importable even in an environment without matplotlib installed at
all.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from numerix.utils.result import MethodResult

_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def _skip(reason: str, console: Console) -> None:
    console.print(f"[yellow]\u26a0 Plot skipped: {reason}[/yellow]")


def _show_or_save(plt: Any, mpl: Any, filename_stub: str, console: Console) -> None:
    """Show interactively if possible; otherwise save a PNG and say so."""
    backend = mpl.get_backend().lower()
    if backend not in _NON_INTERACTIVE_BACKENDS:
        try:
            plt.show()
            return
        except Exception:  # noqa: BLE001 -- fall through to file-save fallback below
            pass

    try:
        from datetime import datetime
        from pathlib import Path

        directory = Path("plots")
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = directory / f"{filename_stub}_{timestamp}.png"
        plt.savefig(path)
        plt.close()
        console.print(f"[yellow]No interactive display available \u2014 saved plot to {path} instead.[/yellow]")
    except Exception as exc:  # noqa: BLE001 -- even the file-save fallback must not crash the app
        plt.close("all")
        _skip(str(exc), console)


# ----------------------------------------------------------------------
# Nonlinear Equations: f(x) curve + root marker
# ----------------------------------------------------------------------

def _nonlinear_range(inputs: dict, root: float) -> tuple[float, float]:
    if "a" in inputs and "b" in inputs:
        lo, hi = sorted((inputs["a"], inputs["b"]))
    elif "x0" in inputs and "x1" in inputs:
        lo, hi = sorted((inputs["x0"], inputs["x1"]))
    elif "x0" in inputs:
        lo, hi = sorted((inputs["x0"], root))
    else:
        lo, hi = root - 1.0, root + 1.0

    lo, hi = min(lo, root), max(hi, root)
    pad = max((hi - lo) * 0.2, 0.5)
    return lo - pad, hi + pad


def plot_nonlinear(result: MethodResult, console: Console | None = None) -> None:
    """f(x) curve, x-axis, and a marker at the root found (§7)."""
    console = console or Console()
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        _skip("matplotlib is not installed.", console)
        return

    try:
        from numerix.utils.parser import parse_function

        f_text = result.inputs.get("f") or result.inputs.get("g")
        if f_text is None:
            _skip("no function found in this result's inputs.", console)
            return
        root = result.solution
        if not isinstance(root, (int, float)) or isinstance(root, bool):
            _skip("this result's solution isn't a single number.", console)
            return

        func = parse_function(f_text, "x")
        lo, hi = _nonlinear_range(result.inputs, root)
        xs = [lo + i * (hi - lo) / 399 for i in range(400)]
        ys = []
        for x in xs:
            try:
                ys.append(func(x))
            except ValueError:
                ys.append(float("nan"))  # leaves a gap in the line instead of crashing

        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=f"f(x) = {f_text}")
        ax.axhline(0, linewidth=0.8, color="gray")
        ax.scatter([root], [0.0], zorder=5, marker="*", s=150, color="red", label=f"root \u2248 {root:.6g}")
        ax.set_title(f"{result.method_name} \u2014 root \u2248 {root:.6g}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linewidth=0.3)
        ax.legend()

        _show_or_save(plt, matplotlib, "nonlinear", console)
    except Exception as exc:  # noqa: BLE001 -- plotting must never crash the app
        _skip(str(exc), console)


# ----------------------------------------------------------------------
# Interpolation: data points + reconstructed interpolation curve
# ----------------------------------------------------------------------

def _interpolation_funcs() -> dict[str, Any]:
    # Imported lazily (not at module level) purely to keep import order
    # simple; these are cheap, ordinary Python imports either way.
    from numerix.core.interpolation import (
        cubic,
        lagrange,
        linear,
        newton_divided_diff,
        newton_gregory_backward,
        newton_gregory_forward,
        quadratic,
    )

    return {
        "Linear Interpolation": linear,
        "Quadratic Interpolation": quadratic,
        "Cubic Interpolation": cubic,
        "Lagrange Interpolation": lagrange,
        "Newton Divided Difference": newton_divided_diff,
        "Newton-Gregory Forward": newton_gregory_forward,
        "Newton-Gregory Backward": newton_gregory_backward,
    }


def plot_interpolation(result: MethodResult, console: Console | None = None) -> None:
    """Data points + the reconstructed interpolation curve (§7).

    The curve is drawn by re-running the *same* interpolation method
    that produced `result` at many sample points across the data
    range -- rather than a second, separate curve-fitting
    implementation living only in the plotting code -- so the plot is
    guaranteed to show exactly the polynomial that produced the
    result, never a subtly different one.
    """
    console = console or Console()
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        _skip("matplotlib is not installed.", console)
        return

    try:
        func = _interpolation_funcs().get(result.method_name)
        if func is None:
            _skip(f"no plot routine for '{result.method_name}'.", console)
            return

        xs = result.inputs["xs"]
        ys = result.inputs["ys"]
        x_target = result.inputs["x_target"]
        y_target = result.solution

        lo, hi = min(xs + [x_target]), max(xs + [x_target])
        pad = max((hi - lo) * 0.1, 0.5)
        lo, hi = lo - pad, hi + pad

        sample_xs = [lo + i * (hi - lo) / 199 for i in range(200)]
        sample_ys = []
        for x in sample_xs:
            try:
                sample_ys.append(func(xs, ys, x).solution)
            except ValueError:
                sample_ys.append(float("nan"))

        fig, ax = plt.subplots()
        ax.plot(sample_xs, sample_ys, label=f"{result.method_name} curve")
        ax.scatter(xs, ys, zorder=5, label="data points")
        ax.scatter(
            [x_target], [y_target], zorder=6, marker="*", s=150, color="red",
            label=f"f({x_target:.6g}) \u2248 {y_target:.6g}",
        )
        ax.set_title(result.method_name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linewidth=0.3)
        ax.legend()

        _show_or_save(plt, matplotlib, "interpolation", console)
    except Exception as exc:  # noqa: BLE001
        _skip(str(exc), console)


# ----------------------------------------------------------------------
# Numerical Integration: f(x) curve + shaded area under the curve
# ----------------------------------------------------------------------

def plot_integration(result: MethodResult, console: Console | None = None) -> None:
    """f(x) curve over a padded range, with `[a, b]` shaded (§7)."""
    console = console or Console()
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        _skip("matplotlib is not installed.", console)
        return

    try:
        from numerix.utils.parser import parse_function

        f_text = result.inputs["f"]
        a = result.inputs["a"]
        b = result.inputs["b"]
        func = parse_function(f_text, "x")

        pad = (b - a) * 0.1 or 0.5
        lo, hi = a - pad, b + pad
        curve_xs = [lo + i * (hi - lo) / 399 for i in range(400)]
        curve_ys = []
        for x in curve_xs:
            try:
                curve_ys.append(func(x))
            except ValueError:
                curve_ys.append(float("nan"))

        shade_xs = [a + i * (b - a) / 199 for i in range(200)]
        shade_ys = [func(x) for x in shade_xs]

        fig, ax = plt.subplots()
        ax.plot(curve_xs, curve_ys, label=f"f(x) = {f_text}")
        ax.fill_between(shade_xs, shade_ys, alpha=0.3, label=f"\u222b f(x) dx \u2248 {result.solution:.6g}")
        ax.axhline(0, linewidth=0.8, color="gray")
        ax.set_title(result.method_name)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linewidth=0.3)
        ax.legend()

        _show_or_save(plt, matplotlib, "integration", console)
    except Exception as exc:  # noqa: BLE001
        _skip(str(exc), console)


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

def plot_result(result: MethodResult, console: Console | None = None) -> None:
    """Render the right plot for `result.category`, or skip gracefully."""
    console = console or Console()
    if result.category == "Nonlinear Equations":
        plot_nonlinear(result, console)
    elif result.category == "Interpolation":
        plot_interpolation(result, console)
    elif result.category == "Numerical Integration":
        plot_integration(result, console)
    else:
        _skip(f"no plot type is defined for '{result.category}'.", console)