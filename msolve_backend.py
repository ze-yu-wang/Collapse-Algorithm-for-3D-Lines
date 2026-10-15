"""Exact rational three-variable interface to the MSolve command-line tool."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


Exponent = tuple[int, ...]
Polynomial = Mapping[Exponent, int | Fraction]
_INTEGER_TOKEN = re.compile(r"-?\d+")
_NONNEGATIVE_INTEGER_TOKEN = re.compile(r"\d+")


class MsolveError(RuntimeError):
    """Base class for explicit MSolve backend failures."""


class MsolveUnavailableError(MsolveError):
    pass


class MsolveCapabilityError(MsolveError):
    pass


class MsolveOutputError(MsolveError):
    pass


class PositiveDimensionalSystemError(MsolveError):
    pass


@dataclass(frozen=True)
class MsolveInfo:
    executable: str
    version: str
    supports_rational_isolating_boxes: bool


@dataclass(frozen=True)
class RealRoot:
    """One MSolve-isolated real root and its exact rational coordinate box.

    These checks establish only the structural validity of the box.  The fact
    that it isolates a root is supplied by MSolve, not proved by this class.
    Integer endpoints are normalized to :class:`~fractions.Fraction`; floats
    and Booleans are rejected because they are not accepted exact coordinates.
    """

    variables: tuple[str, ...]
    box: tuple[tuple[Fraction, Fraction], ...]
    precision_bits: int

    def __post_init__(self) -> None:
        if isinstance(self.variables, (str, bytes)):
            raise TypeError("variables must be an iterable of variable names, not a string")
        variables = tuple(self.variables)
        _validate_variables(variables)

        try:
            raw_box = tuple(self.box)
        except TypeError as error:
            raise TypeError("root box must be an iterable of coordinate intervals") from error
        if len(raw_box) != len(variables):
            raise ValueError(
                f"root box has {len(raw_box)} coordinate intervals; expected {len(variables)}"
            )

        normalized_box: list[tuple[Fraction, Fraction]] = []
        for coordinate_index, raw_interval in enumerate(raw_box):
            try:
                interval = tuple(raw_interval)
            except TypeError as error:
                raise TypeError(
                    f"root box coordinate {coordinate_index} must be a two-endpoint interval"
                ) from error
            if len(interval) != 2:
                raise ValueError(
                    f"root box coordinate {coordinate_index} must have exactly two endpoints"
                )
            lower = _coefficient(interval[0], f"root box coordinate {coordinate_index} lower endpoint")
            upper = _coefficient(interval[1], f"root box coordinate {coordinate_index} upper endpoint")
            if lower > upper:
                raise ValueError(
                    f"root box coordinate {coordinate_index} lower endpoint exceeds upper endpoint"
                )
            normalized_box.append((lower, upper))

        if (
            isinstance(self.precision_bits, bool)
            or not isinstance(self.precision_bits, int)
            or self.precision_bits <= 0
        ):
            raise ValueError("precision_bits must be a positive integer")

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "box", tuple(normalized_box))

    @property
    def exact_midpoint(self) -> tuple[Fraction, ...]:
        return tuple((lower + upper) / 2 for lower, upper in self.box)

    @property
    def approximate(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.exact_midpoint)


def _coefficient(value: int | Fraction, location: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{location} must be an int or Fraction, not {type(value).__name__}")
    return Fraction(value)


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def polynomial_to_msolve(poly: Polynomial, variables: Sequence[str] = ("x", "y", "z")) -> str:
    """Serialize a sparse exact polynomial without converting coefficients to floats."""

    variables = tuple(variables)
    _validate_variables(variables)
    terms: list[tuple[str, str]] = []
    for exponent, raw_coefficient in sorted(poly.items(), key=lambda item: item[0], reverse=True):
        if len(exponent) != len(variables) or any(
            isinstance(power, bool) or not isinstance(power, int) or power < 0
            for power in exponent
        ):
            raise ValueError(f"invalid exponent {exponent!r} for {len(variables)} variables")
        coefficient = _coefficient(raw_coefficient, f"coefficient at {exponent}")
        if coefficient == 0:
            continue
        factors = []
        for variable, power in zip(variables, exponent):
            if power == 1:
                factors.append(variable)
            elif power > 1:
                factors.append(f"{variable}^{power}")
        monomial = "*".join(factors)
        magnitude = abs(coefficient)
        if not monomial:
            body = _fraction_text(magnitude)
        elif magnitude == 1:
            body = monomial
        else:
            body = f"{_fraction_text(magnitude)}*{monomial}"
        terms.append(("-" if coefficient < 0 else "+", body))

    if not terms:
        return "0"
    first_sign, first_body = terms[0]
    return ("-" if first_sign == "-" else "") + first_body + "".join(
        sign + body for sign, body in terms[1:]
    )


def _validate_variables(variables: Sequence[str]) -> None:
    if not variables:
        raise ValueError("at least one variable is required")
    if len(set(variables)) != len(variables):
        raise ValueError("variable names must be unique")
    for variable in variables:
        if not isinstance(variable, str):
            raise TypeError("variable names must be strings")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variable):
            raise ValueError(f"invalid MSolve variable name: {variable!r}")


def inspect_msolve(executable: str = "msolve") -> MsolveInfo:
    """Verify rational real-root boxes with a cached functional probe."""

    resolved = shutil.which(executable)
    if resolved is None:
        raise MsolveUnavailableError(f"MSolve executable not found: {executable!r}")
    return _inspect_msolve_cached(str(Path(resolved).resolve()))


def _read_msolve_version(executable: str) -> str:
    for option in ("-V", "--version"):
        try:
            process = subprocess.run(
                [executable, option], capture_output=True, text=True, check=False, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (process.stdout + process.stderr).strip()
        if process.returncode == 0 and output:
            match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", output)
            return match.group(1) if match else output.splitlines()[0][:200]
    return "unknown"


@lru_cache(maxsize=None)
def _inspect_msolve_cached(resolved_executable: str) -> MsolveInfo:
    """Cache only successful checks; exceptions are not retained by lru_cache."""

    version = _read_msolve_version(resolved_executable)
    probe_input = "x,y,z\n0\n2*x-1,\ny-2,\nz+3\n"
    with tempfile.TemporaryDirectory(prefix="collapse-msolve-probe-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        input_path = temporary_path / "probe.ms"
        output_path = temporary_path / "probe.out"
        input_path.write_text(probe_input, encoding="ascii")
        try:
            process = subprocess.run(
                [
                    resolved_executable,
                    "-f",
                    str(input_path),
                    "-o",
                    str(output_path),
                    "-p",
                    "32",
                    "-t",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except OSError as error:
            raise MsolveUnavailableError(
                f"cannot execute MSolve at {resolved_executable!r}: {error}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise MsolveCapabilityError("MSolve capability probe timed out") from error

        if process.returncode != 0:
            raise MsolveCapabilityError(
                f"MSolve {version} failed the rational-box capability probe with exit code "
                f"{process.returncode}; stdout={process.stdout!r}; stderr={process.stderr!r}"
            )
        output_text = ""
        if output_path.exists():
            output_text = output_path.read_text(encoding="utf-8")
        if not output_text.strip():
            output_text = process.stdout

    try:
        roots = parse_msolve_output(output_text, precision_bits=32)
    except MsolveError as error:
        raise MsolveCapabilityError(
            f"MSolve {version} returned unsupported rational-box output during its capability probe: {error}"
        ) from error
    expected = (Fraction(1, 2), Fraction(2), Fraction(-3))
    if len(roots) != 1 or any(
        not (lower <= value <= upper)
        for value, (lower, upper) in zip(expected, roots[0].box)
    ):
        raise MsolveCapabilityError(
            f"MSolve {version} did not isolate the capability-probe root correctly"
        )
    return MsolveInfo(resolved_executable, version, supports_rational_isolating_boxes=True)


def _tokenize(text: str) -> list[str]:
    """Tokenize the complete output, rejecting every unknown character."""

    tokens: list[str] = []
    position = 0
    punctuation = "[],:/^"
    while position < len(text):
        character = text[position]
        if character.isspace():
            position += 1
            continue
        if character in punctuation:
            tokens.append(character)
            position += 1
            continue
        match = _INTEGER_TOKEN.match(text, position)
        if match:
            token = match.group(0)
            tokens.append(token)
            position = match.end()
            continue
        excerpt = text[position : position + 20].replace("\n", "\\n")
        raise MsolveOutputError(
            f"unrecognized character at offset {position} in MSolve output: {excerpt!r}"
        )
    return tokens


def _parse_fraction(tokens: list[str], position: int) -> tuple[Fraction, int]:
    if position >= len(tokens) or _INTEGER_TOKEN.fullmatch(tokens[position]) is None:
        found = tokens[position] if position < len(tokens) else "<end>"
        raise MsolveOutputError(f"expected integer in MSolve output, found {found!r}")
    numerator = int(tokens[position])
    position += 1
    if position >= len(tokens) or tokens[position] != "/":
        return Fraction(numerator), position
    position += 1
    if position >= len(tokens) or _INTEGER_TOKEN.fullmatch(tokens[position]) is None:
        raise MsolveOutputError("missing rational denominator in MSolve output")
    denominator = int(tokens[position])
    position += 1
    if position < len(tokens) and tokens[position] == "^":
        position += 1
        if position >= len(tokens) or _NONNEGATIVE_INTEGER_TOKEN.fullmatch(tokens[position]) is None:
            raise MsolveOutputError("invalid denominator exponent in MSolve output")
        denominator **= int(tokens[position])
        position += 1
    if denominator == 0:
        raise MsolveOutputError("zero rational denominator in MSolve output")
    return Fraction(numerator, denominator), position


def _parse_list(tokens: list[str], position: int) -> tuple[list[Any], int]:
    if position >= len(tokens) or tokens[position] != "[":
        raise MsolveOutputError("expected '[' in MSolve output")
    position += 1
    result: list[Any] = []
    if position < len(tokens) and tokens[position] == "]":
        return result, position + 1
    while position < len(tokens):
        if tokens[position] == "[":
            item, position = _parse_list(tokens, position)
        else:
            item, position = _parse_fraction(tokens, position)
        result.append(item)
        if position >= len(tokens):
            break
        if tokens[position] == "]":
            return result, position + 1
        if tokens[position] != ",":
            raise MsolveOutputError(
                f"expected ',' or ']' in MSolve output, found {tokens[position]!r}"
            )
        position += 1
        if position >= len(tokens) or tokens[position] == "]":
            raise MsolveOutputError("trailing comma in MSolve output list")
    raise MsolveOutputError("unterminated list in MSolve output")


def parse_msolve_output(
    text: str,
    variables: Sequence[str] = ("x", "y", "z"),
    *,
    precision_bits: int = 128,
) -> list[RealRoot]:
    """Parse MSolve's rational-coefficient real-root output strictly."""

    variables = tuple(variables)
    _validate_variables(variables)
    tokens = _tokenize(text)
    if not tokens:
        raise MsolveOutputError("MSolve produced empty or unrecognized output")
    parsed, position = _parse_list(tokens, 0)
    if position < len(tokens) and tokens[position] == ":":
        position += 1
    if position != len(tokens):
        raise MsolveOutputError("unexpected data after MSolve result")

    if parsed == [Fraction(-1)]:
        return []
    if (
        len(parsed) == 4
        and parsed[0] == Fraction(1)
        and isinstance(parsed[1], Fraction)
        and parsed[2] == Fraction(-1)
        and parsed[3] == []
    ):
        if parsed[1] != len(variables):
            raise MsolveOutputError(
                f"positive-dimensional marker declares {parsed[1]} variables; expected {len(variables)}"
            )
        raise PositiveDimensionalSystemError("MSolve reported a positive-dimensional system")
    if not (
        len(parsed) == 2
        and parsed[0] == Fraction(0)
        and isinstance(parsed[1], list)
        and len(parsed[1]) == 2
        and parsed[1][0] == Fraction(1)
        and isinstance(parsed[1][1], list)
    ):
        raise MsolveOutputError(f"unsupported MSolve output structure: {parsed!r}")

    raw_roots = parsed[1][1]
    roots: list[RealRoot] = []
    for root_index, raw_box in enumerate(raw_roots):
        if not isinstance(raw_box, list) or len(raw_box) != len(variables):
            raise MsolveOutputError(
                f"root {root_index} has {len(raw_box) if isinstance(raw_box, list) else 'non-list'} "
                f"coordinates; expected {len(variables)}"
            )
        box: list[tuple[Fraction, Fraction]] = []
        for coordinate_index, interval in enumerate(raw_box):
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(isinstance(endpoint, Fraction) for endpoint in interval)
            ):
                raise MsolveOutputError(
                    f"root {root_index}, coordinate {coordinate_index} is not a rational interval"
                )
            lower, upper = interval
            if lower > upper:
                raise MsolveOutputError(
                    f"root {root_index}, coordinate {coordinate_index} has reversed interval"
                )
            box.append((lower, upper))
        roots.append(RealRoot(variables, tuple(box), precision_bits))
    return roots


def solve_polynomial_system(
    polynomials: Sequence[Polynomial],
    variables: Sequence[str] = ("x", "y", "z"),
    *,
    executable: str = "msolve",
    precision_bits: int = 128,
    threads: int = 1,
    timeout_seconds: float | None = None,
) -> list[RealRoot]:
    """Return every isolated real root reported by MSolve, with exact boxes.

    This milestone intentionally performs no certified interval evaluation.  A
    positive-dimensional input or an unsupported MSolve interface is reported
    explicitly rather than converted into an empty root list.
    """

    variables = tuple(variables)
    _validate_variables(variables)
    if len(variables) != 3:
        raise ValueError("this direct-3D backend requires exactly three variables")
    polynomials = tuple(polynomials)
    if not polynomials:
        raise ValueError("at least one polynomial is required")
    if not isinstance(precision_bits, int) or precision_bits <= 0:
        raise ValueError("precision_bits must be a positive integer")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    info = inspect_msolve(executable)
    serialized = [polynomial_to_msolve(poly, variables) for poly in polynomials]
    input_text = ",".join(variables) + "\n0\n" + ",\n".join(serialized) + "\n"
    with tempfile.TemporaryDirectory(prefix="collapse-msolve-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        input_path = temporary_path / "system.ms"
        output_path = temporary_path / "roots.out"
        input_path.write_text(input_text, encoding="ascii")
        try:
            process = subprocess.run(
                [
                    info.executable,
                    "-f",
                    str(input_path),
                    "-o",
                    str(output_path),
                    "-p",
                    str(precision_bits),
                    "-t",
                    str(threads),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise MsolveError(f"MSolve timed out after {timeout_seconds} seconds") from error
        except OSError as error:
            raise MsolveUnavailableError(f"failed to execute MSolve: {error}") from error
        if process.returncode != 0:
            raise MsolveError(
                f"MSolve {info.version} failed with exit code {process.returncode}; "
                f"stdout={process.stdout!r}; stderr={process.stderr!r}"
            )
        output_text = ""
        if output_path.exists():
            output_text = output_path.read_text(encoding="utf-8")
        if not output_text.strip():
            output_text = process.stdout
        if not output_text.strip():
            raise MsolveOutputError("MSolve succeeded but produced no output")
    return parse_msolve_output(output_text, variables, precision_bits=precision_bits)


__all__ = [
    "MsolveCapabilityError",
    "MsolveError",
    "MsolveInfo",
    "MsolveOutputError",
    "MsolveUnavailableError",
    "PositiveDimensionalSystemError",
    "RealRoot",
    "inspect_msolve",
    "parse_msolve_output",
    "polynomial_to_msolve",
    "solve_polynomial_system",
]
