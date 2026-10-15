"""File and interactive CLI for the certified adaptive event counter."""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import time
from typing import Any, Sequence

from collapse_events import EventCountResult, FinalDeletionMultiplicityError, count_events
from line_geometry import Line3D
from msolve_backend import MsolveError
from rational_intervals import CertifiedSign
from root_refinement import RootRefinementError


class InputFormatError(ValueError):
    """Raised when input does not describe valid exact rational lines."""


def _rational(value: object, location: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InputFormatError(
            f"{location} must be an integer or rational string, not "
            f"{type(value).__name__}"
        )
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise InputFormatError(
            f"{location} is not a valid rational number: {value!r}"
        ) from error


def _vector(value: object, location: str) -> tuple[Fraction, Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 3:
        raise InputFormatError(
            f"{location} must be an array of exactly three coordinates"
        )
    return tuple(
        _rational(coordinate, f"{location}[{index}]")
        for index, coordinate in enumerate(value)
    )  # type: ignore[return-value]


def parse_lines(data: object) -> tuple[Line3D, ...]:
    """Parse the documented JSON value into exact rational lines."""

    if not isinstance(data, dict):
        raise InputFormatError("top-level JSON value must be an object containing 'lines'")
    if set(data) != {"lines"}:
        missing = "'lines' is required" if "lines" not in data else "unexpected top-level fields"
        raise InputFormatError(missing)
    raw_lines = data["lines"]
    if not isinstance(raw_lines, list):
        raise InputFormatError("'lines' must be an array")

    lines = []
    for index, raw_line in enumerate(raw_lines):
        location = f"lines[{index}]"
        if not isinstance(raw_line, dict) or set(raw_line) != {"point", "direction"}:
            raise InputFormatError(
                f"{location} must be an object containing only 'point' and 'direction'"
            )
        point = _vector(raw_line["point"], f"{location}.point")
        direction = _vector(raw_line["direction"], f"{location}.direction")
        try:
            lines.append(Line3D(point, direction))
        except ValueError as error:
            raise InputFormatError(f"{location}: {error}") from error
    return tuple(lines)


def load_lines(path: str | Path) -> tuple[Line3D, ...]:
    """Load and parse one UTF-8 JSON input file."""

    input_path = Path(path)
    try:
        with input_path.open(encoding="utf-8") as input_file:
            data = json.load(input_file)
    except json.JSONDecodeError as error:
        raise InputFormatError(
            f"invalid JSON in {input_path}: line {error.lineno}, column {error.colno}"
        ) from error
    return parse_lines(data)


def _cross(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def validate_general_position(lines: Sequence[Line3D]) -> None:
    """Check the exact pairwise and directional prerequisites used by the CLI.

    These checks require at least two lines, pairwise skewness, and linearly
    independent directions for every triple.  They do not establish all the
    general-position hypotheses of the mathematical paper.
    """

    lines = tuple(lines)
    if len(lines) < 2:
        raise InputFormatError("at least two lines are required")

    for first, second in combinations(range(len(lines)), 2):
        normal = _cross(lines[first].direction, lines[second].direction)
        displacement = tuple(
            b - a for a, b in zip(lines[first].point, lines[second].point)
        )
        if not any(normal) or _dot(displacement, normal) == 0:
            raise InputFormatError(f"lines[{first}] and lines[{second}] must be skew")

    for first, second, third in combinations(range(len(lines)), 3):
        determinant = _dot(
            lines[first].direction,
            _cross(lines[second].direction, lines[third].direction),
        )
        if determinant == 0:
            raise InputFormatError(
                f"direction vectors of lines[{first}], lines[{second}], and "
                f"lines[{third}] must be linearly independent"
            )


def _read_prompt(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as error:
        raise InputFormatError("input ended before all lines were entered") from error
    except KeyboardInterrupt as error:
        raise InputFormatError("interactive input cancelled") from error


def _prompt_line_count() -> int:
    while True:
        text = _read_prompt("Number of lines (at least 2): ").strip()
        try:
            count = int(text)
        except ValueError:
            print("Please enter an integer number of lines.")
            continue
        if count < 2:
            print("Please enter at least 2 lines.")
            continue
        return count


def _prompt_vector(prompt: str, location: str) -> tuple[Fraction, Fraction, Fraction]:
    while True:
        coordinates = _read_prompt(prompt).split()
        try:
            return _vector(coordinates, location)
        except InputFormatError as error:
            print(f"Invalid coordinates: {error}")


def _interactive_lines() -> tuple[Line3D, ...]:
    """Read exact rational lines interactively from standard input."""

    print(
        "This program counts collapse events for lines in the "
        "three-dimensional farthest Voronoi diagram."
    )
    print("Enter each point and direction as three integers or rational numbers.")
    number_of_lines = _prompt_line_count()
    lines: list[Line3D] = []
    for index in range(number_of_lines):
        print(f"\nLine {index + 1}:")
        point = _prompt_vector(
            "  Point (x y z): ", f"lines[{index}].point"
        )
        while True:
            direction = _prompt_vector(
                "  Direction (dx dy dz): ", f"lines[{index}].direction"
            )
            try:
                line = Line3D(point, direction)
            except ValueError as error:
                print(f"Invalid direction: {error}")
                continue
            lines.append(line)
            break
    return tuple(lines)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if value <= 0 or value == float("inf") or value != value:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        help="input JSON file (omit to enter lines interactively)",
    )
    parser.add_argument("-o", "--output", help="optional detailed JSON result file")
    parser.add_argument(
        "--max-precision",
        "--max-refinement-precision",
        "--maximum-precision-bits",
        dest="maximum_precision_bits",
        type=_positive_int,
        default=1024,
        help="maximum adaptive MSolve precision in bits (default: %(default)s)",
    )
    parser.add_argument(
        "--max-refinement-rounds",
        "--refinement-rounds",
        dest="maximum_refinement_rounds",
        type=_nonnegative_int,
        default=3,
        help=(
            "maximum number of additional refinement precision levels after "
            "the initial 128-bit solve (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--solver-timeout",
        "--timeout",
        type=_positive_float,
        default=None,
        help="per-MSolve-call timeout in seconds (default: no timeout)",
    )
    return parser


def _root_data(root) -> dict[str, Any]:
    return {
        "variables": list(root.variables),
        "precision_bits": root.precision_bits,
        "box": [[str(lower), str(upper)] for lower, upper in root.box],
    }


def _sign_name(sign: CertifiedSign) -> str:
    return {1: "positive", -1: "negative", 0: "zero", None: "uncertain"}[sign]


def _check_data(check) -> dict[str, Any]:
    return {
        "label": check.label,
        "interval": [str(check.interval.lower), str(check.interval.upper)],
        "sign": check.sign,
        "certification": _sign_name(check.sign),
        "precision_bits": check.precision_bits,
    }


def _candidate_data(index: int, candidate) -> dict[str, Any]:
    return {
        "candidate_id": index,
        "family": candidate.family,
        "defining_lines": list(candidate.defining_lines),
        "status": candidate.status,
        "classification": candidate.classification,
        "reason": candidate.reason,
        "final_deletion": candidate.final_deletion,
        "refinement_count": candidate.refinement_count,
        "maximum_precision_used": candidate.maximum_precision_used,
        "original_root": _root_data(candidate.original_root),
        "rounds": [
            {
                "root": _root_data(round_.root),
                "outcome": round_.outcome,
                "reason": round_.reason,
                "checks": [_check_data(check) for check in round_.checks],
            }
            for round_ in candidate.rounds
        ],
    }


def result_data(
    number_of_lines: int,
    result: EventCountResult,
    runtime_seconds: float,
) -> dict[str, Any]:
    diagnostics = result.diagnostics
    identity_left = result.deletions + result.local_minima - result.local_maxima
    identity_right = (
        number_of_lines * number_of_lines
        - number_of_lines
        - 2
        - int(result.final_deletion)
    )
    identity_checked = result.certified
    candidates = [
        _candidate_data(index, candidate)
        for index, candidate in enumerate(diagnostics.candidates)
    ]
    return {
        "version": "2.0",
        "certification_status": "certified" if result.certified else "unresolved",
        "input_lines": number_of_lines,
        "deletions": result.deletions,
        "swaps": result.swaps,
        "local_minima": result.local_minima,
        "local_maxima": result.local_maxima,
        "final_deletion": result.final_deletion,
        "accepted_candidates": len(diagnostics.accepted),
        "rejected_candidates": len(diagnostics.rejected),
        "unresolved_candidates": len(diagnostics.unresolved),
        "solver_failures": len(diagnostics.solver_failures),
        "candidate_refinement_rounds": diagnostics.refinement_count,
        "refinement_solver_calls": diagnostics.refinement_solver_calls,
        "maximum_precision_used": diagnostics.maximum_precision_used,
        "runtime_seconds": runtime_seconds,
        "identity_checked": identity_checked,
        "identity_holds": identity_left == identity_right if identity_checked else None,
        "identity_left": identity_left if identity_checked else None,
        "identity_right": identity_right if identity_checked else None,
        "diagnostics": {
            "triple_systems": diagnostics.triple_systems,
            "quadruple_systems": diagnostics.quadruple_systems,
            "stationary_roots": diagnostics.stationary_roots,
            "vertex_roots": diagnostics.vertex_roots,
            "accepted_candidates": [
                item["candidate_id"]
                for item in candidates
                if item["status"] == "accepted"
            ],
            "rejected_candidates": [
                item["candidate_id"]
                for item in candidates
                if item["status"] == "rejected"
            ],
            "unresolved_candidates": [
                item["candidate_id"]
                for item in candidates
                if item["status"] == "unresolved"
            ],
            "candidates": candidates,
            "system_solves": [
                {
                    "family": solve.family,
                    "defining_lines": list(solve.defining_lines),
                    "precision_bits": solve.precision_bits,
                    "roots": [_root_data(root) for root in solve.roots],
                }
                for solve in diagnostics.system_solves
            ],
            "solver_failures": [
                {
                    "family": failure.family,
                    "defining_lines": list(failure.defining_lines),
                    "precision_bits": failure.precision_bits,
                    "error_type": failure.error_type,
                    "message": failure.message,
                }
                for failure in diagnostics.solver_failures
            ],
            "solver_calls": diagnostics.solver_calls,
            "candidate_refinement_rounds": diagnostics.refinement_count,
            "refinement_solver_calls": diagnostics.refinement_solver_calls,
            "maximum_precision_used": diagnostics.maximum_precision_used,
        },
    }


def format_summary(data: dict[str, Any]) -> str:
    diagnostics = data["diagnostics"]
    if data["identity_checked"]:
        outcome = "holds" if data["identity_holds"] else "FAILS"
        operator = "=" if data["identity_holds"] else "!="
        identity = f"{outcome} ({data['identity_left']} {operator} {data['identity_right']})"
    else:
        identity = "not checked (certification incomplete)"
    return "\n".join(
        (
            "Collapse event counts (version 2.0)",
            f"Deletions: {data['deletions']}",
            f"Swaps: {data['swaps']}",
            f"Local minima: {data['local_minima']}",
            f"Local maxima: {data['local_maxima']}",
            "",
            f"Input lines: {data['input_lines']}",
            f"Special final deletion: {'yes' if data['final_deletion'] else 'no'}",
            f"Certification: {data['certification_status']}",
            f"Counting identity: {identity}",
            f"Runtime: {data['runtime_seconds']:.6f} s",
            "",
            "Certification diagnostics",
            f"Accepted candidates: {len(diagnostics['accepted_candidates'])}",
            f"Rejected candidates: {len(diagnostics['rejected_candidates'])}",
            f"Unresolved candidates: {len(diagnostics['unresolved_candidates'])}",
            f"Solver failures: {len(diagnostics['solver_failures'])}",
            f"Candidate refinement rounds: {diagnostics['candidate_refinement_rounds']}",
            f"Refinement solver calls: {diagnostics['refinement_solver_calls']}",
            f"Maximum precision used: {diagnostics['maximum_precision_used']} bits",
        )
    )


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, separators=(",", ":"), sort_keys=True)
        output_file.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.maximum_precision_bits < 128:
        parser.error("--max-precision must be at least the initial 128 bits")
    try:
        lines = (
            load_lines(arguments.input)
            if arguments.input is not None
            else _interactive_lines()
        )
        validate_general_position(lines)
    except (OSError, InputFormatError) as error:
        if arguments.input is not None:
            parser.error(str(error))
        parser.exit(2, f"Input error: {error}\n")

    started = time.perf_counter()
    try:
        result = count_events(
            lines,
            maximum_precision_bits=arguments.maximum_precision_bits,
            maximum_refinement_rounds=arguments.maximum_refinement_rounds,
            timeout_seconds=arguments.solver_timeout,
        )
    except (MsolveError, RootRefinementError, FinalDeletionMultiplicityError) as error:
        data = {
            "version": "2.0",
            "certification_status": "failed",
            "input_lines": len(lines),
            "runtime_seconds": time.perf_counter() - started,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        print(
            f"Version: 2.0\nCertification: failed\n"
            f"{type(error).__name__}: {error}"
        )
        if arguments.output:
            _write_json(arguments.output, data)
        return 1

    data = result_data(len(lines), result, time.perf_counter() - started)
    print(format_summary(data))
    if arguments.output:
        _write_json(arguments.output, data)
    return 0 if data["identity_checked"] and data["identity_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InputFormatError",
    "build_parser",
    "format_summary",
    "load_lines",
    "main",
    "parse_lines",
    "result_data",
    "validate_general_position",
]
