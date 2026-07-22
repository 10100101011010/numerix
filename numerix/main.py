"""
main.py — Numerix entry point.

Launches the arrow-key category -> method -> input menu (`ui/menu.py`).
All four categories are fully wired, with comparison mode and export
for the nonlinear category, and optional plotting everywhere it
applies -- disable plotting for the whole session with `--no-plot`
(it can also be toggled mid-session from the menu).
"""

from __future__ import annotations

import argparse

from numerix.ui.menu import run_menu


def main() -> None:
    """Entry point (also wired as the `numerix` console script in pyproject.toml)."""
    parser = argparse.ArgumentParser(prog="numerix", description="Classical numerical methods, in your terminal.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="disable plot offers for this session (matplotlib plotting is optional either way)",
    )
    args = parser.parse_args()

    try:
        run_menu(plot_enabled=not args.no_plot)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()