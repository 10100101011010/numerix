# Numerix

Classical numerical methods, in your terminal.

[![Tests](https://github.com/10100101011010/numerix/actions/workflows/test.yml/badge.svg)](https://github.com/<OWNER>/numerix/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<!--
  TODO before publishing: replace <OWNER> above with the real GitHub
  username/org, and drop a real terminal capture in docs/demo.gif
  (e.g. via `asciinema` + `agg`, or a screen recording converted to
  GIF) so the line below actually renders something.
-->
![Numerix demo](docs/demo.gif)

## What is this

Numerix is a terminal application implementing **23 classical numerical
methods** across four categories — nonlinear equations, linear systems,
interpolation, and numerical integration — entirely from first
principles. No `numpy.linalg`, no library quadrature, no library
interpolators: every method is hand-written on plain Python data
structures, so the point isn't just the answer, it's watching the
method work.

Every result — regardless of category or method — renders through one
consistent format: an input recap, the full iteration/step table, the
final result, a stats footer, and a converged/warning indicator. Every
category also has a **Compare Methods** mode that runs several methods
on the same problem side by side, and any result (single or comparison)
can be exported to `results/` as CSV or JSON. Plotting is available
wherever it's meaningful (root-finding, integration, interpolation) and
can be toggled off from Settings or via `--no-plot`.

Function input is plain math text (`x**3 - x - 2`, `sin(x) - x/2`),
parsed safely with `sympy` — never `eval`.

## Installation

### Option 1 — download a prebuilt binary (no Python required)

Grab the binary for your OS from the [Releases page](https://github.com/<OWNER>/numerix/releases)
— look for the asset matching `numerix-<version>-<os>-<arch>`.

**Windows**
1. Download `numerix-<version>-windows-x64.exe`.
2. Run it — double-click, or from a terminal:
   ```powershell
   .\numerix-<version>-windows-x64.exe
   ```
3. Windows SmartScreen may flag it as unrecognized (it's an unsigned
   binary) — click **More info → Run anyway**.

**macOS**
1. Download `numerix-<version>-macos-arm64`.
2. Make it executable and clear the Gatekeeper quarantine flag that
   macOS adds to anything downloaded from a browser:
   ```bash
   chmod +x numerix-<version>-macos-arm64
   xattr -d com.apple.quarantine numerix-<version>-macos-arm64
   ```
3. Run it:
   ```bash
   ./numerix-<version>-macos-arm64
   ```

**Linux**
1. Download `numerix-<version>-linux-x64`.
2. Make it executable and run it:
   ```bash
   chmod +x numerix-<version>-linux-x64
   ./numerix-<version>-linux-x64
   ```

### Option 2 — `pip install` (all three OSes, fallback to the binaries)

Requires Python 3.11+.

```bash
git clone https://github.com/<OWNER>/numerix.git
cd numerix
pip install .
numerix
```

`pip install .` uses the console-script entry point declared in
`pyproject.toml`, so the `numerix` command is on your `PATH` afterward.
Pass `--no-plot` to disable plot offers for the session:

```bash
numerix --no-plot
```

## The methods

23 methods across four categories:

**Nonlinear Equations** (5)
Bisection · Regula Falsi · Fixed Point · Newton-Raphson · Secant

**Linear Systems** (6)
Gaussian Elimination · Gauss-Jordan · Matrix Inverse · LU Decomposition
· Jacobi · Gauss-Seidel

**Interpolation** (7)
Linear Interpolation · Quadratic Interpolation · Cubic Interpolation ·
Lagrange Interpolation · Newton Divided Difference · Newton-Gregory
Forward · Newton-Gregory Backward

**Numerical Integration** (5)
Rectangle Rule (left/right) · Midpoint Rule · Trapezoidal Rule ·
Simpson's 1/3 Rule · Simpson's 3/8 Rule

## Building from source

```bash
git clone https://github.com/<OWNER>/numerix.git
cd numerix
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m numerix.main
```

Run the test suite:

```bash
pytest
```

Build your own standalone binary (must be built on the OS you're
targeting — PyInstaller doesn't cross-compile):

```bash
pip install pyinstaller
pyinstaller --onefile --name numerix numerix/main.py
```

The binary lands in `dist/` (`dist/numerix.exe` on Windows,
`dist/numerix` on macOS/Linux).

## License

MIT — see [LICENSE](LICENSE).
