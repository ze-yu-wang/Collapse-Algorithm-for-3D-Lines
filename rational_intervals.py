"""Exact rational interval arithmetic for certified polynomial signs.

Every operation returns a guaranteed rational enclosure; no endpoint is ever
converted to floating point.  Polynomial evaluation on a box therefore
encloses the value at every point of that box.  Repeated occurrences of a
variable are evaluated independently, so the enclosure can be conservative
through dependency overestimation.  The implementation intentionally favors a
small, auditable surface over clever interval optimizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Mapping, Sequence


Exponent = tuple[int, ...]
Polynomial = Mapping[Exponent, int | Fraction]
CertifiedSign = Literal[-1, 0, 1] | None


def _fraction(value: int | Fraction, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be an int or Fraction, not {type(value).__name__}")
    return Fraction(value)


@dataclass(frozen=True)
class RationalInterval:
    """A closed interval with exact rational endpoints and enclosure arithmetic."""

    lower: Fraction
    upper: Fraction

    def __init__(self, lower: int | Fraction, upper: int | Fraction | None = None) -> None:
        exact_lower = _fraction(lower, "lower endpoint")
        exact_upper = exact_lower if upper is None else _fraction(upper, "upper endpoint")
        if exact_lower > exact_upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lower", exact_lower)
        object.__setattr__(self, "upper", exact_upper)

    @property
    def is_singleton(self) -> bool:
        return self.lower == self.upper

    def __add__(self, other: object) -> RationalInterval:
        other_interval = as_interval(other)
        return RationalInterval(
            self.lower + other_interval.lower,
            self.upper + other_interval.upper,
        )

    def __radd__(self, other: object) -> RationalInterval:
        return self + other

    def __sub__(self, other: object) -> RationalInterval:
        other_interval = as_interval(other)
        return RationalInterval(
            self.lower - other_interval.upper,
            self.upper - other_interval.lower,
        )

    def __rsub__(self, other: object) -> RationalInterval:
        return as_interval(other) - self

    def __neg__(self) -> RationalInterval:
        return RationalInterval(-self.upper, -self.lower)

    def __mul__(self, other: object) -> RationalInterval:
        other_interval = as_interval(other)
        products = (
            self.lower * other_interval.lower,
            self.lower * other_interval.upper,
            self.upper * other_interval.lower,
            self.upper * other_interval.upper,
        )
        return RationalInterval(min(products), max(products))

    def __rmul__(self, other: object) -> RationalInterval:
        return self * other

    def __pow__(self, exponent: int) -> RationalInterval:
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 0:
            raise ValueError("interval exponent must be a nonnegative integer")
        if exponent == 0:
            return RationalInterval(1)
        if exponent % 2:
            return RationalInterval(self.lower**exponent, self.upper**exponent)
        if self.lower <= 0 <= self.upper:
            return RationalInterval(0, max(abs(self.lower), abs(self.upper)) ** exponent)
        endpoint_powers = (self.lower**exponent, self.upper**exponent)
        return RationalInterval(min(endpoint_powers), max(endpoint_powers))


def as_interval(value: object) -> RationalInterval:
    if isinstance(value, RationalInterval):
        return value
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        return RationalInterval(value)
    raise TypeError(f"expected RationalInterval, int, or Fraction, not {type(value).__name__}")


def evaluate_polynomial_interval(
    polynomial: Polynomial,
    box: Sequence[RationalInterval | tuple[int | Fraction, int | Fraction]],
) -> RationalInterval:
    """Enclose the polynomial value at every point of ``box`` exactly.

    The returned rational interval is guaranteed to contain the full image,
    but may be wider than that image because ordinary interval evaluation does
    not track dependencies between repeated uses of a coordinate.
    """

    intervals = tuple(
        value if isinstance(value, RationalInterval) else RationalInterval(*value)
        for value in box
    )
    result = RationalInterval(0)
    for exponent, raw_coefficient in polynomial.items():
        if len(exponent) != len(intervals):
            raise ValueError(
                f"exponent {exponent!r} has dimension {len(exponent)}; expected {len(intervals)}"
            )
        if any(isinstance(power, bool) or not isinstance(power, int) or power < 0 for power in exponent):
            raise ValueError(f"invalid exponent {exponent!r}")
        coefficient = _fraction(raw_coefficient, f"coefficient at {exponent}")
        term = RationalInterval(coefficient)
        for coordinate, power in zip(intervals, exponent):
            term *= coordinate**power
        result += term
    return result


def certified_sign(interval: RationalInterval) -> CertifiedSign:
    """Return a proven sign, or ``None`` when the interval does not decide it.

    Excluding zero certifies a positive or negative sign.  Merely containing
    zero does not establish that a polynomial vanishes at an isolated root;
    zero is returned only for the exact singleton ``[0, 0]``.
    """

    if not isinstance(interval, RationalInterval):
        raise TypeError("certified_sign requires a RationalInterval")
    if interval.lower > 0:
        return 1
    if interval.upper < 0:
        return -1
    if interval.lower == interval.upper == 0:
        return 0
    return None


__all__ = [
    "CertifiedSign",
    "RationalInterval",
    "as_interval",
    "certified_sign",
    "evaluate_polynomial_interval",
]
