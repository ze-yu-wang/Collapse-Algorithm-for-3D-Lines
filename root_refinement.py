"""Adaptive MSolve refinement with exact, containment-certified root matching."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Mapping, Sequence

from msolve_backend import RealRoot, solve_polynomial_system


Exponent = tuple[int, ...]
Polynomial = Mapping[Exponent, int | Fraction]
Solver = Callable[..., list[RealRoot]]


class RootRefinementError(RuntimeError):
    """Base class for failures that make root correspondence uncertifiable."""


class InconsistentRootCountError(RootRefinementError):
    pass


class NonUniqueRootMatchError(RootRefinementError):
    pass


class RootPreservationError(RootRefinementError):
    pass


@dataclass(frozen=True)
class RefinementConfig:
    initial_precision_bits: int = 128
    maximum_precision_bits: int = 1024
    maximum_refinement_rounds: int = 3
    timeout_seconds: float | None = None
    executable: str = "msolve"
    threads: int = 1

    def __post_init__(self) -> None:
        for name in ("initial_precision_bits", "maximum_precision_bits", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_precision_bits < self.initial_precision_bits:
            raise ValueError("maximum_precision_bits must be at least initial_precision_bits")
        if (
            isinstance(self.maximum_refinement_rounds, bool)
            or not isinstance(self.maximum_refinement_rounds, int)
            or self.maximum_refinement_rounds < 0
        ):
            raise ValueError("maximum_refinement_rounds must be a nonnegative integer")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")

    @property
    def precisions(self) -> tuple[int, ...]:
        values = [self.initial_precision_bits]
        current = self.initial_precision_bits
        for _ in range(self.maximum_refinement_rounds):
            if current >= self.maximum_precision_bits:
                break
            current = min(current * 2, self.maximum_precision_bits)
            values.append(current)
        return tuple(values)


def box_contains(outer: RealRoot, inner: RealRoot) -> bool:
    """Return whether ``inner`` is coordinatewise contained in ``outer`` exactly."""

    return (
        outer.variables == inner.variables
        and len(outer.box) == len(inner.box)
        and all(
            outer_lower <= inner_lower <= inner_upper <= outer_upper
            for (outer_lower, outer_upper), (inner_lower, inner_upper) in zip(
                outer.box, inner.box
            )
        )
    )


def match_refined_roots(
    original_roots: Sequence[RealRoot], refined_roots: Sequence[RealRoot]
) -> tuple[RealRoot, ...]:
    """Align refined roots to originals using a uniquely certified containment.

    Containment is stronger than box overlap: because each original box
    isolates exactly one root of the same system, a refined isolating box
    inside it necessarily preserves that root.
    """

    original_roots = tuple(original_roots)
    refined_roots = tuple(refined_roots)
    if len(original_roots) != len(refined_roots):
        raise InconsistentRootCountError(
            f"root count changed from {len(original_roots)} to {len(refined_roots)}"
        )
    matches = tuple(
        tuple(index for index, refined in enumerate(refined_roots) if box_contains(original, refined))
        for original in original_roots
    )
    reverse_matches = tuple(
        tuple(index for index, original in enumerate(original_roots) if box_contains(original, refined))
        for refined in refined_roots
    )
    if any(len(indices) > 1 for indices in matches) or any(
        len(indices) > 1 for indices in reverse_matches
    ):
        raise NonUniqueRootMatchError(
            f"root-box containment is nonunique: original-to-refined={matches!r}, "
            f"refined-to-original={reverse_matches!r}"
        )
    if any(len(indices) == 0 for indices in matches) or any(
        len(indices) == 0 for indices in reverse_matches
    ):
        raise RootPreservationError(
            f"a refined box is not uniquely nested in its original isolating box: "
            f"original-to-refined={matches!r}, refined-to-original={reverse_matches!r}"
        )
    return tuple(refined_roots[indices[0]] for indices in matches)


def _system_key(polynomials: Sequence[Polynomial]) -> tuple[tuple[tuple[tuple[int, ...], int, int], ...], ...]:
    return tuple(
        tuple(
            sorted(
                (tuple(exponent), Fraction(coefficient).numerator, Fraction(coefficient).denominator)
                for exponent, coefficient in polynomial.items()
                if coefficient != 0
            )
        )
        for polynomial in polynomials
    )


class AdaptiveRootRefiner:
    """Solve and refine systems while caching each system/precision invocation."""

    def __init__(self, config: RefinementConfig = RefinementConfig(), *, solver: Solver = solve_polynomial_system):
        if not isinstance(config, RefinementConfig):
            raise TypeError("config must be a RefinementConfig")
        self.config = config
        self._solver = solver
        self._cache: dict[tuple[object, int], tuple[RealRoot, ...] | Exception] = {}
        self._aligned_cache: dict[
            tuple[object, int, tuple[RealRoot, ...]], tuple[RealRoot, ...]
        ] = {}
        self.solver_calls = 0

    def solve(self, polynomials: Sequence[Polynomial], precision_bits: int) -> tuple[RealRoot, ...]:
        polynomials = tuple(polynomials)
        key = (_system_key(polynomials), precision_bits)
        cached = self._cache.get(key)
        if isinstance(cached, Exception):
            raise cached
        if cached is not None:
            return cached
        self.solver_calls += 1
        try:
            roots = tuple(
                self._solver(
                    polynomials,
                    executable=self.config.executable,
                    precision_bits=precision_bits,
                    threads=self.config.threads,
                    timeout_seconds=self.config.timeout_seconds,
                )
            )
        except Exception as error:
            self._cache[key] = error
            raise
        self._cache[key] = roots
        return roots

    def initial_roots(self, polynomials: Sequence[Polynomial]) -> tuple[RealRoot, ...]:
        return self.solve(polynomials, self.config.initial_precision_bits)

    def roots_at_precision(
        self,
        polynomials: Sequence[Polynomial],
        original_roots: Sequence[RealRoot],
        precision_bits: int,
    ) -> tuple[RealRoot, ...]:
        """Return roots at ``precision_bits`` aligned to the original ordering."""

        if precision_bits == self.config.initial_precision_bits:
            return tuple(original_roots)
        system_key = _system_key(tuple(polynomials))
        ordered_original_roots = tuple(original_roots)
        key = (system_key, precision_bits, ordered_original_roots)
        if key not in self._aligned_cache:
            refined = self.solve(polynomials, precision_bits)
            self._aligned_cache[key] = match_refined_roots(ordered_original_roots, refined)
        return self._aligned_cache[key]


__all__ = [
    "AdaptiveRootRefiner",
    "InconsistentRootCountError",
    "NonUniqueRootMatchError",
    "RefinementConfig",
    "RootPreservationError",
    "RootRefinementError",
    "box_contains",
    "match_refined_roots",
]
