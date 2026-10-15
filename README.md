# Collapse Event Counter for 3D Lines

A Python implementation for counting the **deletion**, **swap**, **local minimum**, and **local maximum** events arising in the collapse process of the Euclidean farthest Voronoi diagram of lines in three dimensions. The program takes a finite set of lines as input and returns certified event counts; it does not construct the full Voronoi diagram.

The geometric definitions and theory are given in the accompanying paper:

> Evanthia Papadopoulou, Martin Suderland, and Zeyu Wang. **A Collapse Process for Farthest Voronoi Diagrams of Lines in Three Dimensions.** [arXiv:2609.29865](https://arxiv.org/abs/2609.29865), 2026.  
> Also appeared in ISAAC 2026: forthcoming.

## Requirements

- Python **3.10+** (no third-party Python packages required).
- [MSolve](https://msolve.lip6.fr/) installed as an executable named `msolve` on your `PATH`. The interface has been tested with **MSolve 0.10.0**; it checks the installed solver's rational-root-box capability at runtime. Installation instructions are available in the [MSolve repository](https://github.com/algebraic-solving/msolve).

## Usage

From the repository root, run the included four-line example:

```bash
python main.py --input examples/four_lines.json
```

To enter lines interactively, run:

```bash
python main.py
```

Each line is specified by a point and a nonzero direction vector, with three coordinates each. Coordinates must be exact integers or rational values (e.g., `-2`, `3/4`). For JSON inputs, rational values must be strings, not JSON floating-point numbers.

To write detailed certification diagnostics, including isolating boxes and sign checks, use:

```bash
python main.py --input examples/four_lines.json --output result.json
```

Run `python main.py --help` for precision, refinement-round, and solver-timeout options.

## Example

The included [`examples/four_lines.json`](examples/four_lines.json) contains:

```json
{
  "lines": [
    {"point": [-2, 3, 0], "direction": [2, 3, -3]},
    {"point": [-2, 1, -2], "direction": [-3, 2, -1]},
    {"point": [-3, -1, 2], "direction": [-2, -1, 1]},
    {"point": [0, -2, 3], "direction": [-3, -3, 2]}
  ]
}
```

The certified result for this configuration is **0 deletions, 0 swaps, 10 local minima, and 0 local maxima**. No special final deletion occurs; the counting identity holds. The computation completes with 128-bit root-isolating boxes.

**Validation and release scope.** This repository intentionally contains only the computational kernel and a reproducible example. Development-time testing and benchmarking were carried out separately and are not part of this release. The included example demonstrates one configuration; it does not exercise every event type.

## Python API

```python
from main import load_lines, validate_general_position
from collapse_events import count_events

lines = load_lines("examples/four_lines.json")
validate_general_position(lines)
result = count_events(lines)

print(result.counts)       # (deletions, swaps, local_minima, local_maxima)
print(result.certified)    # True if no roots or solver failures remain unresolved
print(result.final_deletion)
```

Lines may also be constructed directly with `line_geometry.Line3D(point, direction)`. The `count_events` function accepts optional solver/refinement settings and returns an `EventCountResult` containing the counts and detailed diagnostics.

## Implementation structure

The kernel separates exact geometry from root isolation and event classification:

| File | Responsibility |
| --- | --- |
| `line_geometry.py` | Exact rational line geometry and sparse multivariate polynomial construction. |
| `msolve_backend.py` | Polynomial-system serialization, MSolve invocation, and parsing of rational root-isolating boxes. |
| `rational_intervals.py` | Exact interval arithmetic and certified polynomial-sign evaluation. |
| `root_refinement.py` | Adaptive refinement and correspondence of isolating boxes across precisions. |
| `collapse_events.py` | Candidate enumeration, global-farthest filtering, event classification, and certification diagnostics. |
| `main.py` | Input validation, command-line interface, and optional JSON output. |

The implementation reuses constructed distance polynomials and other shared expressions, and caches polynomial-system solves during refinement. It enumerates one stationary system per line triple and one equidistance system per line quadruple, rather than constructing the full diagram. No output-sensitive runtime bound is claimed for this kernel.

## Method and scope

For each triple of lines, the program constructs polynomial equations for stationary points of the common-distance function along their trisector; for each quadruple, it constructs equidistance equations. **MSolve** isolates their real roots. Exact rational interval evaluation then certifies global-farthest membership and classifies the relevant events, with adaptive root-box refinement when needed.

Writing `D_i` for the squared-distance polynomial of line `i`, the stationary system for three lines `i, j, k` is

```text
F = D_i - D_j
G = D_i - D_k
T = gradient(F) x gradient(G)
F = G = gradient(D_i) · T = 0
```

The next directional derivative along `T` distinguishes local minima from local maxima at regular stationary roots. Four-line candidates satisfy three equal-distance equations; exact gradient-determinant signs then distinguish deletion and swap events, including the special final deletion case.

The computation assumes the **general-position hypotheses of the paper**. The input validator checks pairwise skewness and linear independence of every triple of directions, but does not verify all of those hypotheses. Certification of polynomial signs is exact, conditional on the correctness of the isolating boxes returned by MSolve.

## Interpreting certification

A `certified` status means the program completed its candidate enumeration with no unresolved candidate roots or solver failures. An `unresolved` status means the available computations do not establish complete event counts; a `failed` status indicates an error that prevented the computation from completing. The counting identity is checked only when certification succeeds, and the command-line program returns a nonzero exit status if certification or that check does not succeed.

These are **conditional** guarantees: the result relies on MSolve's root-isolation correctness and on the mathematical general-position assumptions stated above. The included example is not a substitute for those assumptions or for independent validation.
