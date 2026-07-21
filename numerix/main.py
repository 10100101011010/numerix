"""
main.py — Numerix entry point.

Launches the arrow-key category -> method -> input menu (`ui/menu.py`,
Phase 5). The nonlinear category is fully wired; the other three
categories are listed but not yet implemented (Phases 6-8).
"""

from __future__ import annotations

from numerix.ui.menu import run_menu


def main() -> None:
    """Entry point (also wired as the `numerix` console script in pyproject.toml)."""
    try:
        run_menu()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()