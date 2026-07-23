"""
main.py — Numerix entry point.

Shows a one-time ASCII startup banner, then launches the arrow-key
category -> method -> input menu (`ui/menu.py`). All four categories
are fully wired, each with its own comparison mode and export, and
optional plotting everywhere it applies -- disable plotting for the
whole session with `--no-plot` (it can also be toggled mid-session
from the menu's Settings entry).
"""

from __future__ import annotations

import argparse

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from numerix.ui.display import render_error
from numerix.ui.menu import run_menu

# Generated once via pyfiglet (font="standard") and pinned here as a
# literal so startup has no extra dependency or runtime cost.
_BANNER_LINES = [
    " _   _ _   _ __  __ _____ ____  _____  __",
    "| \\ | | | | |  \\/  | ____|  _ \\|_ _\\ \\/ /",
    "|  \\| | | | | |\\/| |  _| | |_) || | \\  / ",
    "| |\\  | |_| | |  | | |___|  _ < | | /  \\ ",
    "|_| \\_|\\___/|_|  |_|_____|_| \\_\\___/_/\\_\\",
]
_BANNER_ART = "\n".join(f"  {line}" for line in _BANNER_LINES)
_TAGLINE = "Classical numerical methods, in your terminal."


def _print_startup_banner() -> None:
    """Show a one-time ASCII banner before the main menu, styled with the
    same cyan-panel treatment `ui/display.py` uses for method banners.

    Built as a `rich.text.Text` object rather than an f-string with
    inline `[bold cyan]...[/bold cyan]` markup: the banner art ends in
    a literal backslash, and Rich's markup parser treats a backslash
    immediately before `[` as an escape for a literal bracket -- so a
    markup string here would swallow the closing tag and print
    "[/bold cyan]" as literal text instead of applying it as a style.
    `Text` never parses its content as markup, so this is safe
    regardless of what characters the art contains.
    """
    console = Console()
    content = Text(_BANNER_ART, style="bold cyan")
    content.append("\n\n")
    content.append(_TAGLINE, style="dim")
    console.print(Panel(Align.center(content), border_style="cyan"))


def main() -> None:
    """Entry point (also wired as the `numerix` console script in pyproject.toml)."""
    parser = argparse.ArgumentParser(prog="numerix", description="Classical numerical methods, in your terminal.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="disable plot offers for this session (matplotlib plotting is optional either way)",
    )
    args = parser.parse_args()

    _print_startup_banner()

    try:
        run_menu(plot_enabled=not args.no_plot)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
    except Exception as exc:  # noqa: BLE001 -- absolute last resort, per §8
        # Every dispatch inside run_menu's own loop is already wrapped by
        # _run_safely (ui/menu.py), so in practice nothing should reach
        # here. This exists purely so a bug in the loop's own control
        # flow -- outside any of those wrapped dispatches -- still ends
        # in one friendly line instead of a raw traceback on exit.
        render_error(f"unexpected error: {exc}", Console())


if __name__ == "__main__":
    main()