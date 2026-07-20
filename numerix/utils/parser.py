"""
utils/parser.py — safe string -> callable function parsing (§6.1).

Users type math as plain text, e.g. "x**3 - x - 2" or "sin(x) - x/2".
This module NEVER uses bare `eval`. It goes through `sympy.sympify`
(via `parse_expr`) against a restricted, explicit symbol table, then
`sympy.lambdify` to produce a fast numeric callable. It also exposes
symbolic differentiation (`sympy.diff`) so Newton-Raphson never has
to ask the user to type a derivative by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ----------------------------------------------------------------------
# Restricted symbol table
# ----------------------------------------------------------------------
# Only these names resolve when parsing a user-typed expression. We
# pass global_dict={} explicitly below, which suppresses sympy's
# default behavior of importing its entire namespace (`from sympy
# import *`) into scope — without that, things like `Integral`,
# `Matrix`, or `oo` would be silently callable from user input. Any
# bare identifier that isn't one of these names, or one of the
# declared variables, is caught after parsing and rejected with a
# friendly ParseError instead of being auto-promoted to a new symbol.

_ALLOWED_FUNCTIONS: dict[str, object] = {
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
    "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
    "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
    "sqrt": sympy.sqrt, "Abs": sympy.Abs, "abs": sympy.Abs,
    "floor": sympy.floor, "ceiling": sympy.ceiling,
}

_ALLOWED_CONSTANTS: dict[str, object] = {
    "pi": sympy.pi, "e": sympy.E, "E": sympy.E,
}

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_symbol_cache: dict[str, sympy.Symbol] = {}


class ParseError(ValueError):
    """Raised when a user-typed expression can't be safely parsed.

    Always carries a short, friendly message suitable for direct
    display to the end user (per §7/§8: never a raw traceback).
    """


@dataclass
class ParsedFunction:
    """A user-typed expression, its sympy form, and a fast callable."""

    text: str                     # the exact string the user typed
    variables: tuple[str, ...]    # e.g. ("x",)
    expr: sympy.Expr              # parsed sympy expression
    func: Callable[..., float]    # numeric callable, same argument order as `variables`

    def __call__(self, *args: float) -> float:
        """Evaluate the function, converting failure modes into a plain ValueError."""
        try:
            result = self.func(*args)
        except ZeroDivisionError as exc:
            raise ValueError(f"division by zero evaluating '{self.text}'") from exc
        except ValueError as exc:
            raise ValueError(f"'{self.text}' is undefined at this input ({exc})") from exc
        except OverflowError as exc:
            raise ValueError(f"'{self.text}' overflowed at this input") from exc

        if isinstance(result, complex):
            raise ValueError(f"'{self.text}' produced a complex value, not a real number")
        return float(result)

    def derivative(self, variable: str | None = None) -> "ParsedFunction":
        """Return a new ParsedFunction for d/d(variable) of this expression.

        Used by Newton-Raphson (§6.2) so the user never has to type
        f'(x) themselves — the derivative is derived symbolically via
        `sympy.diff`, per §6.1.
        """
        var_name = variable or self.variables[0]
        symbol = _symbol_for(var_name)
        d_expr = sympy.diff(self.expr, symbol)
        d_func = sympy.lambdify(self._symbols(), d_expr, modules=["math"])
        return ParsedFunction(
            text=f"d/d{var_name}[{self.text}]",
            variables=self.variables,
            expr=d_expr,
            func=d_func,
        )

    def _symbols(self) -> tuple[sympy.Symbol, ...]:
        return tuple(_symbol_for(name) for name in self.variables)


def _symbol_for(name: str) -> sympy.Symbol:
    if name not in _symbol_cache:
        _symbol_cache[name] = sympy.Symbol(name, real=True)
    return _symbol_cache[name]


def parse_function(text: str, variables: str | tuple[str, ...] = "x") -> ParsedFunction:
    """Safely parse a user-typed expression like "x**3 - x - 2".

    Args:
        text: The raw string the user typed.
        variables: The variable name(s) the expression is allowed to
            use, e.g. "x", or ("x", "y"). Any other bare identifier
            found in the expression is treated as a parse error rather
            than silently promoted to a new free symbol.

    Returns:
        A ParsedFunction wrapping the sympy expression and a fast
        numeric callable.

    Raises:
        ParseError: on empty input, syntax errors, or use of any
            identifier outside the allowed symbol table.
    """
    if isinstance(variables, str):
        variables = (variables,)

    stripped = text.strip()
    if not stripped:
        raise ParseError("function input cannot be empty")

    var_symbols = {name: _symbol_for(name) for name in variables}
    local_dict: dict[str, object] = {**_ALLOWED_FUNCTIONS, **_ALLOWED_CONSTANTS, **var_symbols}

    try:
        expr = parse_expr(
            stripped,
            local_dict=local_dict,
            global_dict={
                "Integer": sympy.Integer,
                "Float": sympy.Float,
                "Rational": sympy.Rational,
                "Symbol": sympy.Symbol,
            },
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except (SyntaxError, TypeError, sympy.SympifyError) as exc:
        raise ParseError(f"could not parse '{text}' as a math expression") from exc

    allowed_symbol_names = set(variables)
    stray = {str(s) for s in expr.free_symbols} - allowed_symbol_names
    if stray:
        names = ", ".join(sorted(stray))
        raise ParseError(
            f"unknown name(s) in '{text}': {names} "
            f"(only {', '.join(variables)} and standard math functions are allowed)"
        )

    try:
        func = sympy.lambdify(tuple(var_symbols.values()), expr, modules=["math"])
    except Exception as exc:  # pragma: no cover - lambdify is very permissive
        raise ParseError(f"could not build a callable for '{text}'") from exc

    return ParsedFunction(text=stripped, variables=tuple(variables), expr=expr, func=func)