"""
utils/result.py — Numerix core data contract (§5).

Every algorithm in `numerix.core`, regardless of category, returns a
`MethodResult`. Every renderer and every export path elsewhere in the
app consumes only this one shape — do not invent a parallel result
type or a separate export schema anywhere else in the codebase.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MethodResult:
    """The single, shared result contract for every numerical method.

    Attributes:
        method_name: Human-readable method name, e.g. "Newton-Raphson".
        category: One of "Nonlinear Equations", "Linear Systems",
            "Interpolation", "Numerical Integration".
        inputs: Everything the user provided, echoed back verbatim.
        iterations: One dict per row. Empty list if non-iterative in
            the convergence-loop sense (e.g. Simpson's rule) — per
            §6.5 those methods still populate this with one row per
            subinterval, so it is rarely truly empty in practice.
        solution: Root, solution vector, interpolated value, or
            integral value.
        approx_error: Final approximate/relative error; None if not
            applicable.
        n_iterations: 0 if non-iterative.
        converged: False also covers "max iterations hit without
            meeting tolerance".
        warning: e.g. "matrix not diagonally dominant — convergence
            not guaranteed". None if no warning applies.
        exec_time_ms: Wall-clock execution time of the algorithm, in
            milliseconds.
    """

    method_name: str
    category: str
    inputs: dict
    iterations: list[dict]
    solution: float | list | dict
    approx_error: float | None
    n_iterations: int
    converged: bool
    warning: str | None = None
    exec_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain dict of its dataclass fields.

        This is the single source of truth for both CSV and JSON
        export — neither format hand-rolls a separate schema.
        """
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this result to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_fallback)

    def to_csv(self) -> str:
        """Serialize this result to a CSV string.

        `MethodResult` mixes scalar metadata with a tabular
        `iterations` list, so the CSV has two sections, both drawn
        directly from the dataclass's own fields:

        1. A `field,value` metadata block for every non-iteration
           field (`inputs` flattened to `inputs.<key>` rows).
        2. A blank separator line, then the iteration table with one
           column per key seen across all iteration rows (methods may
           report different columns per row, e.g. an extra warning
           column only on the final row).
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow(["field", "value"])
        writer.writerow(["method_name", self.method_name])
        writer.writerow(["category", self.category])
        for key, value in self.inputs.items():
            writer.writerow([f"inputs.{key}", _flatten_for_csv(value)])
        writer.writerow(["solution", _flatten_for_csv(self.solution)])
        writer.writerow(["approx_error", self.approx_error])
        writer.writerow(["n_iterations", self.n_iterations])
        writer.writerow(["converged", self.converged])
        writer.writerow(["warning", self.warning or ""])
        writer.writerow(["exec_time_ms", self.exec_time_ms])

        if self.iterations:
            writer.writerow([])
            fieldnames: list[str] = []
            for row in self.iterations:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            dict_writer = csv.DictWriter(buf, fieldnames=fieldnames, restval="")
            dict_writer.writeheader()
            for row in self.iterations:
                dict_writer.writerow(row)

        return buf.getvalue()

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def save(self, directory: str | Path, fmt: str) -> Path:
        """Write this result to `directory` as a timestamped file.

        `fmt` must be "json" or "csv". Returns the path written to.
        Creates `directory` if it doesn't already exist (the app's
        `results/` folder is gitignored, not guaranteed to exist).
        """
        if fmt not in ("json", "csv"):
            raise ValueError(f"unsupported export format: {fmt!r} (expected 'json' or 'csv')")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self.method_name.lower().replace(" ", "_").replace("-", "_")
        path = directory / f"{slug}_{timestamp}.{fmt}"

        content = self.to_json() if fmt == "json" else self.to_csv()
        path.write_text(content, encoding="utf-8")
        return path


def _json_fallback(value: Any) -> Any:
    """Best-effort conversion for values `json.dumps` can't handle natively."""
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    return str(value)


def _flatten_for_csv(value: Any) -> Any:
    """Render list/dict values as a compact JSON string for CSV metadata rows."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=_json_fallback)
    return value