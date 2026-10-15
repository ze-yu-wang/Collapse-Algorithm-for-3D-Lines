"""Exact direct-3D polynomial constructions for lines in R^3.

Polynomials are sparse dictionaries mapping exponent triples ``(i, j, k)``
to exact :class:`fractions.Fraction` coefficients of
``x**i * y**j * z**k``.  Missing monomials have coefficient zero, and stored
zero coefficients are removed.  Exponents are nonnegative integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, NamedTuple, Sequence


Exponent = tuple[int, int, int]
Polynomial = dict[Exponent, Fraction]
VectorPolynomial = tuple[Polynomial, Polynomial, Polynomial]
Rational3 = tuple[Fraction, Fraction, Fraction]


def _rational(value: int | Fraction, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be an int or Fraction, not {type(value).__name__}")
    return Fraction(value)


def _rational3(values: Iterable[int | Fraction], name: str) -> Rational3:
    values = tuple(values)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    return tuple(_rational(value, f"{name}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


@dataclass(frozen=True)
class Line3D:
    """A line represented by an exact rational point and direction vector.

    Integer inputs are normalized to :class:`~fractions.Fraction`, so both
    fields retain exact rational data.  The direction must be nonzero; its
    length and scaling are otherwise immaterial to the represented line.
    """

    point: Rational3
    direction: Rational3

    def __init__(
        self,
        point: Iterable[int | Fraction],
        direction: Iterable[int | Fraction],
    ) -> None:
        exact_point = _rational3(point, "point")
        exact_direction = _rational3(direction, "direction")
        if all(component == 0 for component in exact_direction):
            raise ValueError("line direction must be nonzero")
        object.__setattr__(self, "point", exact_point)
        object.__setattr__(self, "direction", exact_direction)


def _clean(poly: Mapping[Exponent, int | Fraction]) -> Polynomial:
    result: Polynomial = {}
    for exponent, coefficient in poly.items():
        if (
            not isinstance(exponent, tuple)
            or len(exponent) != 3
            or any(
                isinstance(power, bool) or not isinstance(power, int) or power < 0
                for power in exponent
            )
        ):
            raise ValueError(f"invalid exponent triple: {exponent!r}")
        exact = _rational(coefficient, f"coefficient at {exponent}")
        if exact:
            result[exponent] = result.get(exponent, Fraction(0)) + exact
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def constant(value: int | Fraction) -> Polynomial:
    exact = _rational(value, "constant")
    return {} if exact == 0 else {(0, 0, 0): exact}


def variable(index: int) -> Polynomial:
    if isinstance(index, bool) or index not in (0, 1, 2):
        raise ValueError("variable index must be 0, 1, or 2")
    exponent = [0, 0, 0]
    exponent[index] = 1
    return {tuple(exponent): Fraction(1)}  # type: ignore[dict-item]


def add_polynomials(*polynomials: Mapping[Exponent, int | Fraction]) -> Polynomial:
    result: Polynomial = {}
    for poly in polynomials:
        for exponent, coefficient in _clean(poly).items():
            result[exponent] = result.get(exponent, Fraction(0)) + coefficient
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def scale_polynomial(poly: Mapping[Exponent, int | Fraction], scalar: int | Fraction) -> Polynomial:
    exact = _rational(scalar, "scalar")
    return _clean({exponent: coefficient * exact for exponent, coefficient in _clean(poly).items()})


def subtract_polynomials(
    left: Mapping[Exponent, int | Fraction],
    right: Mapping[Exponent, int | Fraction],
) -> Polynomial:
    return add_polynomials(left, scale_polynomial(right, -1))


def multiply_polynomials(
    left: Mapping[Exponent, int | Fraction],
    right: Mapping[Exponent, int | Fraction],
) -> Polynomial:
    result: Polynomial = {}
    for left_exponent, left_coefficient in _clean(left).items():
        for right_exponent, right_coefficient in _clean(right).items():
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            result[exponent] = result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def derivative(poly: Mapping[Exponent, int | Fraction], variable_index: int) -> Polynomial:
    if isinstance(variable_index, bool) or variable_index not in (0, 1, 2):
        raise ValueError("variable index must be 0, 1, or 2")
    result: Polynomial = {}
    for exponent, coefficient in _clean(poly).items():
        power = exponent[variable_index]
        if power:
            derived_exponent = list(exponent)
            derived_exponent[variable_index] -= 1
            result[tuple(derived_exponent)] = coefficient * power
    return result


def gradient(poly: Mapping[Exponent, int | Fraction]) -> VectorPolynomial:
    return tuple(derivative(poly, index) for index in range(3))  # type: ignore[return-value]


def dot_polynomial_vectors(left: Sequence[Polynomial], right: Sequence[Polynomial]) -> Polynomial:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("polynomial vectors must have three components")
    return add_polynomials(*(multiply_polynomials(a, b) for a, b in zip(left, right)))


def cross_polynomial_vectors(left: Sequence[Polynomial], right: Sequence[Polynomial]) -> VectorPolynomial:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("polynomial vectors must have three components")
    return (
        subtract_polynomials(multiply_polynomials(left[1], right[2]), multiply_polynomials(left[2], right[1])),
        subtract_polynomials(multiply_polynomials(left[2], right[0]), multiply_polynomials(left[0], right[2])),
        subtract_polynomials(multiply_polynomials(left[0], right[1]), multiply_polynomials(left[1], right[0])),
    )


def evaluate_polynomial(poly: Mapping[Exponent, int | Fraction], point: Sequence[object]):
    """Evaluate a polynomial pointwise at three exact or approximate values.

    Approximate coordinates are intentionally permitted for diagnostics.  A
    resulting sign is not certified; use exact interval evaluation from
    :mod:`rational_intervals` when a proof of sign is required.
    """

    if len(point) != 3:
        raise ValueError("point must contain exactly three coordinates")
    return sum(
        coefficient * point[0] ** exponent[0] * point[1] ** exponent[1] * point[2] ** exponent[2]
        for exponent, coefficient in _clean(poly).items()
    )


def polynomial_degree(poly: Mapping[Exponent, int | Fraction]) -> int:
    """Return total degree, with the zero polynomial assigned degree -1."""

    cleaned = _clean(poly)
    return max((sum(exponent) for exponent in cleaned), default=-1)


def squared_distance_polynomial(line: Line3D) -> Polynomial:
    """Return the exact squared Euclidean distance to ``line``.

    For point ``X``, line point ``P``, and direction ``v``, the polynomial is
    ``||X-P||^2 - ((X-P) dot v)^2 / (v dot v)``.  All coefficients remain
    rational, and the result is independent of the scale of ``v``.
    """

    coordinates = (variable(0), variable(1), variable(2))
    displacement = tuple(
        add_polynomials(coordinate, constant(-offset))
        for coordinate, offset in zip(coordinates, line.point)
    )
    direction_norm_squared = sum(component * component for component in line.direction)
    displacement_norm_squared = dot_polynomial_vectors(displacement, displacement)
    projection_numerator = add_polynomials(
        *(scale_polynomial(component, direction) for component, direction in zip(displacement, line.direction))
    )
    return subtract_polynomials(
        displacement_norm_squared,
        scale_polynomial(multiply_polynomials(projection_numerator, projection_numerator), 1 / direction_norm_squared),
    )


def _require_lines(lines: Sequence[Line3D], count: int) -> tuple[Line3D, ...]:
    lines = tuple(lines)
    if len(lines) != count:
        raise ValueError(f"expected exactly {count} lines, got {len(lines)}")
    if not all(isinstance(line, Line3D) for line in lines):
        raise TypeError("all inputs must be Line3D instances")
    return lines


def four_line_vertex_system(lines: Sequence[Line3D]) -> tuple[Polynomial, Polynomial, Polynomial]:
    """Construct the four-line equidistance equations directly in R^3.

    If ``Di`` is squared distance to line ``i``, the returned equations are
    ``D1-D2``, ``D1-D3``, and ``D1-D4``.  Their common zeros are precisely the
    points at equal squared distance from all four lines.
    """

    line1, line2, line3, line4 = _require_lines(lines, 4)
    distances = tuple(squared_distance_polynomial(line) for line in (line1, line2, line3, line4))
    return tuple(subtract_polynomials(distances[0], distance) for distance in distances[1:])  # type: ignore[return-value]


class StationarySystem(NamedTuple):
    """Exact polynomials defining stationary points of a three-line trisector.

    With ``Di`` the squared distance to line ``i``, ``F = D1-D2`` and
    ``G = D1-D3`` define the trisector.  ``T = grad(F) cross grad(G)`` is its
    tangent field, ``C = grad(D1) dot T`` is the stationarity equation along
    that field, and ``H = grad(C) dot T`` is the next directional derivative.
    The solver equations are exactly ``(F, G, C)``.
    """

    F: Polynomial
    G: Polynomial
    T: VectorPolynomial
    C: Polynomial
    H: Polynomial

    @property
    def equations(self) -> tuple[Polynomial, Polynomial, Polynomial]:
        return self.F, self.G, self.C


def three_line_stationary_system(lines: Sequence[Line3D]) -> StationarySystem:
    """Construct ``F``, ``G``, tangent ``T``, stationary equation ``C``, and ``H``.

    Here ``F=D1-D2``, ``G=D1-D3``, ``T=grad(F) cross grad(G)``,
    ``C=grad(D1) dot T``, and ``H=grad(C) dot T``.
    """

    line1, line2, line3 = _require_lines(lines, 3)
    distance1, distance2, distance3 = (
        squared_distance_polynomial(line) for line in (line1, line2, line3)
    )
    F = subtract_polynomials(distance1, distance2)
    G = subtract_polynomials(distance1, distance3)
    T = cross_polynomial_vectors(gradient(F), gradient(G))
    C = dot_polynomial_vectors(gradient(distance1), T)
    H = dot_polynomial_vectors(gradient(C), T)
    return StationarySystem(F=F, G=G, T=T, C=C, H=H)


__all__ = [
    "Exponent",
    "Line3D",
    "Polynomial",
    "StationarySystem",
    "VectorPolynomial",
    "add_polynomials",
    "cross_polynomial_vectors",
    "derivative",
    "dot_polynomial_vectors",
    "evaluate_polynomial",
    "four_line_vertex_system",
    "gradient",
    "multiply_polynomials",
    "polynomial_degree",
    "scale_polynomial",
    "squared_distance_polynomial",
    "subtract_polynomials",
    "three_line_stationary_system",
]
