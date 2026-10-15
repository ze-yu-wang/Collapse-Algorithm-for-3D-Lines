"""Certified adaptive counting of collapse events for lines in three dimensions.

For exact rational input lines, the algorithm:

1. constructs the squared-distance polynomials ``D_i`` exactly;
2. enumerates the stationary systems of every line triple and the equidistance
   systems of every line quadruple;
3. asks MSolve for rational isolating boxes for all real roots;
4. certifies global-farthest membership with interval signs of ``D_i-D_j``;
5. classifies surviving roots by the geometric criteria of the paper;
6. refines an isolating box whenever a sign needed by the classification is
   still uncertain; and
7. returns the event counts together with the complete certification history.

For defining lines ``i < j < k``, let ``F = D_i-D_j``, ``G = D_i-D_k``,
``T = grad(F) cross grad(G)``, ``C = grad(D_i) dot T``, and
``H = grad(C) dot T``.  The stationary system is ``F = G = C = 0``.  At a
regular stationary root, positive ``H`` denotes a local minimum and negative
``H`` a local maximum.

For ``i < j < k < l``, the four determinant polynomials are formed from the
distance gradients in increasing index order.  Their alternating combination
``eta`` and their correspondingly alternating signs determine how many sides
of a four-line vertex are FVD-positive: two gives a swap, three an ordinary
deletion, and four the unique special final deletion.

Under the mathematical hypotheses of the counting theorem, the convention is
``n_d + n_m - n_M = n*n - n - 2 - int(final_deletion)``.  Certified interval
signs make each implemented sign decision rigorous; they do not by themselves
establish all general-position hypotheses or MSolve's root-isolation guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Literal, Sequence

from rational_intervals import (
    CertifiedSign,
    RationalInterval,
    certified_sign,
    evaluate_polynomial_interval,
)
from line_geometry import (
    Line3D,
    Polynomial,
    StationarySystem,
    add_polynomials,
    cross_polynomial_vectors,
    dot_polynomial_vectors,
    gradient,
    scale_polynomial,
    squared_distance_polynomial,
    subtract_polynomials,
)
from msolve_backend import MsolveError, RealRoot
from root_refinement import AdaptiveRootRefiner, RefinementConfig, RootRefinementError


CandidateFamily = Literal["stationary", "vertex"]
CandidateStatus = Literal["accepted", "rejected", "unresolved"]
EventKind = Literal["local_minimum", "local_maximum", "deletion", "swap"]
DecisionOutcome = Literal["accepted", "rejected", "unresolved", "refine"]


@dataclass(frozen=True)
class CertifiedSignCheck:
    """One exact interval evaluation and its certified sign, if determined."""

    label: str
    interval: RationalInterval
    sign: CertifiedSign
    precision_bits: int


@dataclass(frozen=True)
class CertificationRound:
    """All sign checks performed on one isolating box during certification."""

    root: RealRoot
    checks: tuple[CertifiedSignCheck, ...]
    outcome: DecisionOutcome
    reason: str | None


@dataclass(frozen=True)
class CandidateRecord:
    """Certification result and history for one isolated candidate root.

    Accepted candidates have every required sign certified and an event type
    assigned.  Rejected candidates have certified evidence excluding the root
    from the relevant events.  Unresolved candidates encountered a genuine
    degeneracy, exhausted refinement, or could not complete safe refinement.
    """

    family: CandidateFamily
    defining_lines: tuple[int, ...]
    original_root: RealRoot
    rounds: tuple[CertificationRound, ...]
    status: CandidateStatus
    classification: EventKind | None
    reason: str | None
    final_deletion: bool = False

    @property
    def root(self) -> RealRoot:
        """The most refined box actually used for this candidate."""

        return self.rounds[-1].root if self.rounds else self.original_root

    @property
    def checks(self) -> tuple[CertifiedSignCheck, ...]:
        """The checks made in the last completed certification round."""

        return self.rounds[-1].checks if self.rounds else ()

    @property
    def root_history(self) -> tuple[RealRoot, ...]:
        """The isolating boxes on which classification was actually evaluated."""

        return tuple(round_.root for round_ in self.rounds)

    @property
    def refinement_count(self) -> int:
        """Number of completed higher-precision certification rounds.

        A failed refinement solver call or failed root correspondence is not a
        completed round and is instead represented by failure diagnostics.
        """

        return max(0, len(self.rounds) - 1)

    @property
    def maximum_precision_used(self) -> int:
        """Highest precision of a box on which this candidate was evaluated."""

        return max(
            (round_.root.precision_bits for round_ in self.rounds),
            default=self.original_root.precision_bits,
        )


@dataclass(frozen=True)
class SystemSolveRecord:
    """Raw roots returned for one system at one requested precision."""

    family: CandidateFamily
    defining_lines: tuple[int, ...]
    precision_bits: int
    roots: tuple[RealRoot, ...]


@dataclass(frozen=True)
class SolverFailure:
    """A solver or safe-correspondence failure retained for diagnostics."""

    family: CandidateFamily
    defining_lines: tuple[int, ...]
    precision_bits: int
    error_type: str
    message: str


@dataclass(frozen=True)
class CountDiagnostics:
    """System-level and candidate-level evidence supporting an event count."""

    triple_systems: int
    quadruple_systems: int
    stationary_roots: int
    vertex_roots: int
    candidates: tuple[CandidateRecord, ...]
    system_solves: tuple[SystemSolveRecord, ...]
    solver_failures: tuple[SolverFailure, ...]
    solver_calls: int

    @property
    def accepted(self) -> tuple[CandidateRecord, ...]:
        """Candidates whose required signs and event classifications are certified."""

        return tuple(
            candidate for candidate in self.candidates if candidate.status == "accepted"
        )

    @property
    def rejected(self) -> tuple[CandidateRecord, ...]:
        """Candidates excluded by certified geometric evidence."""

        return tuple(
            candidate for candidate in self.candidates if candidate.status == "rejected"
        )

    @property
    def unresolved(self) -> tuple[CandidateRecord, ...]:
        """Candidates not classified because certification could not be completed."""

        return tuple(
            candidate for candidate in self.candidates if candidate.status == "unresolved"
        )

    @property
    def refinement_count(self) -> int:
        """Total completed higher-precision candidate rounds.

        Several candidates from one polynomial system may share a single
        cached MSolve call, so this is intentionally not a solver-call count.
        """

        return sum(candidate.refinement_count for candidate in self.candidates)

    @property
    def refinement_solver_calls(self) -> int:
        """Number of recorded successful system solves above the initial 128 bits."""

        return sum(solve.precision_bits > 128 for solve in self.system_solves)

    @property
    def maximum_precision_used(self) -> int:
        """Highest precision among successfully recorded system solves."""

        return max((solve.precision_bits for solve in self.system_solves), default=0)


@dataclass(frozen=True)
class EventCountResult:
    """Certified event totals and the diagnostics from which they were derived."""

    deletions: int
    swaps: int
    local_minima: int
    local_maxima: int
    diagnostics: CountDiagnostics
    final_deletion: bool = False

    @property
    def counts(self) -> tuple[int, int, int, int]:
        return self.deletions, self.swaps, self.local_minima, self.local_maxima

    @property
    def certified(self) -> bool:
        """Whether enumeration completed without unresolved roots or failures.

        This guarantee is conditional on MSolve's root isolation and on the
        mathematical input assumptions of the event-counting method.
        """

        return not self.diagnostics.unresolved and not self.diagnostics.solver_failures


@dataclass(frozen=True)
class _Decision:
    """Internal result of applying all currently decidable geometric tests."""

    outcome: DecisionOutcome
    classification: EventKind | None
    reason: str | None
    checks: tuple[CertifiedSignCheck, ...]
    final_deletion: bool = False


def _determinant_polynomial(
    first: Sequence[Polynomial],
    second: Sequence[Polynomial],
    third: Sequence[Polynomial],
) -> Polynomial:
    """Return ``first dot (second cross third)`` as an exact polynomial."""

    return dot_polynomial_vectors(first, cross_polynomial_vectors(second, third))


class _PolynomialCache:
    """Exact per-run polynomials shared by systems and candidate checks.

    Each squared distance and its gradient is constructed once.  Ordered
    distance differences and system polynomials are then memoized lazily.
    Determinants are shared across quadruples by their increasing line triple;
    this increasing order fixes their orientation, while vertex classification
    applies the required alternating signs explicitly.
    """

    def __init__(self, lines: Sequence[Line3D]) -> None:
        self.distances = tuple(squared_distance_polynomial(line) for line in lines)
        self.distance_gradients = tuple(gradient(distance) for distance in self.distances)
        self._distance_differences: dict[tuple[int, int], Polynomial] = {}
        self._determinants: dict[tuple[int, int, int], Polynomial] = {}
        self._stationary_systems: dict[tuple[int, int, int], StationarySystem] = {}
        self._vertex_equation_systems: dict[
            tuple[int, int, int, int], tuple[Polynomial, Polynomial, Polynomial]
        ] = {}

    @property
    def line_count(self) -> int:
        return len(self.distances)

    def distance_difference(self, first: int, second: int) -> Polynomial:
        """Return the cached ordered difference ``D_first - D_second``."""

        key = (first, second)
        if key not in self._distance_differences:
            self._distance_differences[key] = subtract_polynomials(
                self.distances[first], self.distances[second]
            )
        return self._distance_differences[key]

    def stationary_system(self, indices: tuple[int, int, int]) -> StationarySystem:
        """Return cached ``F``, ``G``, ``T``, ``C``, and ``H`` for a line triple.

        These are the same constructions as
        :func:`line_geometry.three_line_stationary_system`, assembled here from
        cached distances and gradients.
        """

        if indices not in self._stationary_systems:
            first, second, third = indices
            F = self.distance_difference(first, second)
            G = self.distance_difference(first, third)
            T = cross_polynomial_vectors(gradient(F), gradient(G))
            C = dot_polynomial_vectors(self.distance_gradients[first], T)
            H = dot_polynomial_vectors(gradient(C), T)
            self._stationary_systems[indices] = StationarySystem(
                F=F, G=G, T=T, C=C, H=H
            )
        return self._stationary_systems[indices]

    def vertex_equations(
        self, indices: tuple[int, int, int, int]
    ) -> tuple[Polynomial, Polynomial, Polynomial]:
        """Return cached ``D_i-D_j``, ``D_i-D_k``, and ``D_i-D_l`` equations."""

        if indices not in self._vertex_equation_systems:
            first, second, third, fourth = indices
            self._vertex_equation_systems[indices] = (
                self.distance_difference(first, second),
                self.distance_difference(first, third),
                self.distance_difference(first, fourth),
            )
        return self._vertex_equation_systems[indices]

    def determinant(self, indices: tuple[int, int, int]) -> Polynomial:
        """Return the determinant of three gradients in increasing index order.

        Sorting is intentional: cached determinants have one canonical
        orientation.  Callers must apply any permutation or alternating sign
        required by the geometric formula.
        """

        key = tuple(sorted(indices))
        if len(set(key)) != 3:
            raise ValueError("determinant requires three distinct line indices")
        if key not in self._determinants:
            self._determinants[key] = _determinant_polynomial(
                *(self.distance_gradients[index] for index in key)
            )
        return self._determinants[key]

    def quadruple_determinants(
        self, indices: tuple[int, int, int, int]
    ) -> tuple[Polynomial, Polynomial, Polynomial, Polynomial]:
        """Return ``(det_ijk, det_ijl, det_ikl, det_jkl)`` for ``i<j<k<l``."""

        first, second, third, fourth = indices
        return (
            self.determinant((first, second, third)),
            self.determinant((first, second, fourth)),
            self.determinant((first, third, fourth)),
            self.determinant((second, third, fourth)),
        )


def _check(label: str, polynomial: Polynomial, root: RealRoot) -> CertifiedSignCheck:
    """Evaluate one polynomial on ``root.box`` and retain its exact sign evidence."""

    interval = evaluate_polynomial_interval(polynomial, root.box)
    return CertifiedSignCheck(label, interval, certified_sign(interval), root.precision_bits)


def _global_farthest_decision(
    root: RealRoot,
    defining_lines: tuple[int, ...],
    number_of_lines: int,
    distance_difference: Callable[[int, int], Polynomial],
) -> tuple[DecisionOutcome, str | None, tuple[CertifiedSignCheck, ...]]:
    """Certify whether a defining-line root lies on the global farthest diagram.

    The defining lines have equal distance at a candidate root.  Using their
    first line as anchor, every nondefining comparison must have
    ``D_anchor-D_other > 0``.  A negative comparison proves another line is
    farther and rejects immediately.  Exact equality records a degeneracy and
    an interval containing zero requests refinement, but both cases continue
    through later lines so that a subsequent negative can still reject.  Thus
    the returned checks are precisely the comparisons actually evaluated.
    """

    anchor = defining_lines[0]
    defining = set(defining_lines)
    checks: list[CertifiedSignCheck] = []
    exact_equality = False
    uncertain = False
    for line_index in range(number_of_lines):
        if line_index in defining:
            continue
        check = _check(
            f"D[{anchor}]-D[{line_index}]",
            distance_difference(anchor, line_index),
            root,
        )
        checks.append(check)
        if check.sign == -1:
            return (
                "rejected",
                "a nondefining line is certified farther",
                tuple(checks),
            )
        if check.sign == 0:
            exact_equality = True
        elif check.sign is None:
            uncertain = True

    if exact_equality:
        return (
            "unresolved",
            "exact nondefining distance equality (degenerate candidate)",
            tuple(checks),
        )
    if uncertain:
        return "refine", "global farthest sign is not yet certified", tuple(checks)
    return "accepted", None, tuple(checks)


def _stationary_decision(
    root: RealRoot,
    defining_lines: tuple[int, int, int],
    stationary_system: StationarySystem,
    number_of_lines: int,
    distance_difference: Callable[[int, int], Polynomial],
) -> _Decision:
    """Certify global membership, trisector regularity, and the sign of ``H``."""

    outcome, reason, checks = _global_farthest_decision(
        root, defining_lines, number_of_lines, distance_difference
    )
    if outcome != "accepted":
        return _Decision(outcome, None, reason, checks)

    tangent_checks = tuple(
        _check(f"T[{index}]", polynomial, root)
        for index, polynomial in enumerate(stationary_system.T)
    )
    checks += tangent_checks
    if not any(check.sign in (-1, 1) for check in tangent_checks):
        if all(check.sign == 0 for check in tangent_checks):
            return _Decision(
                "unresolved",
                None,
                "trisector tangent is exactly zero (degenerate candidate)",
                checks,
            )
        return _Decision("refine", None, "trisector regularity is not yet certified", checks)

    h_check = _check("H", stationary_system.H, root)
    checks += (h_check,)
    if h_check.sign is None:
        return _Decision("refine", None, "H sign is not yet certified", checks)
    if h_check.sign == 0:
        return _Decision(
            "unresolved",
            None,
            "H is exactly zero (degenerate stationary point)",
            checks,
        )
    classification: EventKind = "local_minimum" if h_check.sign == 1 else "local_maximum"
    return _Decision("accepted", classification, None, checks)


def _eta_polynomial(determinants: Sequence[Polynomial]) -> Polynomial:
    """Construct ``eta = det_ijk - det_ijl + det_ikl - det_jkl`` exactly."""

    return add_polynomials(
        determinants[0],
        scale_polynomial(determinants[1], -1),
        determinants[2],
        scale_polynomial(determinants[3], -1),
    )


def _vertex_decision(
    root: RealRoot,
    defining_lines: tuple[int, int, int, int],
    determinants: Sequence[Polynomial],
    eta: Polynomial,
    number_of_lines: int,
    distance_difference: Callable[[int, int], Polynomial],
) -> _Decision:
    """Certify a four-line vertex and classify its FVD-positive sides.

    After global-farthest membership, all four ordered determinants and
    ``eta`` must have nonzero certified signs.  The oriented side signs are
    ``(det_ijk, -det_ijl, det_ikl, -det_jkl)``; a side is FVD-positive exactly
    when its sign agrees with ``eta``.  Counts two, three, and four classify a
    swap, ordinary deletion, and special final deletion respectively.
    """

    outcome, reason, checks = _global_farthest_decision(
        root, defining_lines, number_of_lines, distance_difference
    )
    if outcome != "accepted":
        return _Decision(outcome, None, reason, checks)

    determinant_checks = tuple(
        _check(f"det[{index}]", polynomial, root)
        for index, polynomial in enumerate(determinants)
    )
    checks += determinant_checks
    if any(check.sign == 0 for check in determinant_checks):
        return _Decision(
            "unresolved", None, "a determinant is exactly zero (degenerate vertex)", checks
        )
    if any(check.sign is None for check in determinant_checks):
        return _Decision("refine", None, "determinant signs are not yet certified", checks)

    eta_check = _check("eta", eta, root)
    checks += (eta_check,)
    if eta_check.sign is None:
        return _Decision("refine", None, "eta sign is not yet certified", checks)
    if eta_check.sign == 0:
        return _Decision("unresolved", None, "eta is exactly zero (degenerate vertex)", checks)

    determinant_signs = tuple(check.sign for check in determinant_checks)
    assert all(sign in (-1, 1) for sign in determinant_signs)
    assert eta_check.sign in (-1, 1)
    side_count = sum(
        sign * eta_check.sign > 0
        for sign in (
            determinant_signs[0],
            -determinant_signs[1],
            determinant_signs[2],
            -determinant_signs[3],
        )
    )
    if side_count not in (2, 3, 4):
        return _Decision(
            "rejected",
            None,
            f"four-line root has {side_count} certified FVD-positive sides, not 2, 3, or 4",
            checks,
        )
    final_deletion = side_count == 4
    classification: EventKind = "swap" if side_count == 2 else "deletion"
    return _Decision(
        "accepted",
        classification,
        (
            "special final deletion: four certified FVD-positive sides"
            if final_deletion
            else None
        ),
        checks,
        final_deletion,
    )


class FinalDeletionMultiplicityError(RuntimeError):
    """Raised when more than one certified four-positive-side deletion exists."""

    def __init__(self, candidates: Sequence[CandidateRecord]):
        self.candidates = tuple(candidates)
        locations = ", ".join(
            f"lines={candidate.defining_lines}, box={candidate.root.box!r}"
            for candidate in self.candidates
        )
        super().__init__(
            f"detected {len(self.candidates)} certified final deletions; at most one is permitted "
            f"({locations})"
        )


def _failure(
    family: CandidateFamily,
    defining_lines: tuple[int, ...],
    precision_bits: int,
    error: Exception,
) -> SolverFailure:
    """Convert an exception into the stable diagnostic representation."""

    return SolverFailure(
        family, defining_lines, precision_bits, type(error).__name__, str(error)
    )


def _certify_root(
    *,
    family: CandidateFamily,
    defining_lines: tuple[int, ...],
    polynomials: Sequence[Polynomial],
    original_roots: Sequence[RealRoot],
    root_index: int,
    refiner: AdaptiveRootRefiner,
    decision_function: Callable[[RealRoot], _Decision],
    register_solve: Callable[
        [CandidateFamily, tuple[int, ...], int, Sequence[RealRoot]], None
    ],
    register_failure: Callable[
        [CandidateFamily, tuple[int, ...], int, Exception], None
    ],
) -> CandidateRecord:
    """Adaptively certify one root while preserving every completed round.

    Classification starts on the original isolating box and advances through
    the configured precisions only after a ``refine`` decision.  The refiner
    proves root correspondence by exact containment.  At higher precision the
    raw solver output is recorded before that proof, so diagnostics retain the
    returned boxes even when matching fails.  Acceptance, certified rejection,
    degeneracy, solver failure, and exhausted resources all stop immediately.
    """

    original_root = original_roots[root_index]
    rounds: list[CertificationRound] = []
    latest_decision: _Decision | None = None
    for precision_bits in refiner.config.precisions:
        try:
            # Register the raw higher-precision output before attempting a
            # correspondence proof, so even unsafe/unmatchable boxes remain
            # available in diagnostics.
            if precision_bits != refiner.config.initial_precision_bits:
                raw_roots = refiner.solve(polynomials, precision_bits)
                register_solve(family, defining_lines, precision_bits, raw_roots)
            roots = refiner.roots_at_precision(polynomials, original_roots, precision_bits)
            if precision_bits == refiner.config.initial_precision_bits:
                register_solve(family, defining_lines, precision_bits, roots)
            root = roots[root_index]
        except (MsolveError, RootRefinementError) as error:
            register_failure(family, defining_lines, precision_bits, error)
            reason = (
                f"certification stopped at {precision_bits} bits: "
                f"{type(error).__name__}: {error}"
            )
            return CandidateRecord(
                family,
                defining_lines,
                original_root,
                tuple(rounds),
                "unresolved",
                None,
                reason,
            )

        latest_decision = decision_function(root)
        rounds.append(
            CertificationRound(
                root,
                latest_decision.checks,
                latest_decision.outcome,
                latest_decision.reason,
            )
        )
        if latest_decision.outcome != "refine":
            status: CandidateStatus = latest_decision.outcome
            return CandidateRecord(
                family,
                defining_lines,
                original_root,
                tuple(rounds),
                status,
                latest_decision.classification,
                latest_decision.reason,
                latest_decision.final_deletion,
            )

    assert latest_decision is not None
    reason = (
        f"{latest_decision.reason}; refinement limits reached at "
        f"{rounds[-1].root.precision_bits} bits after {len(rounds) - 1} refinement rounds"
    )
    return CandidateRecord(
        family,
        defining_lines,
        original_root,
        tuple(rounds),
        "unresolved",
        None,
        reason,
    )


def count_events_v2(
    lines: Sequence[Line3D],
    *,
    msolve_executable: str = "msolve",
    maximum_precision_bits: int = 1024,
    maximum_refinement_rounds: int = 3,
    threads: int = 1,
    timeout_seconds: float | None = None,
    _solver=None,
) -> EventCountResult:
    """Count certified collapse events for a sequence of exact rational lines.

    Three-line stationary roots and four-line equidistance roots are enumerated
    explicitly.  Every isolated root is either classified, rejected by a
    certified comparison, or retained as unresolved with its diagnostic
    history.  The result does not silently discard solver or refinement
    failures.
    """

    lines = tuple(lines)
    if not all(isinstance(line, Line3D) for line in lines):
        raise TypeError("all inputs must be Line3D instances")
    config = RefinementConfig(
        initial_precision_bits=128,
        maximum_precision_bits=maximum_precision_bits,
        maximum_refinement_rounds=maximum_refinement_rounds,
        timeout_seconds=timeout_seconds,
        executable=msolve_executable,
        threads=threads,
    )
    refiner = (
        AdaptiveRootRefiner(config)
        if _solver is None
        else AdaptiveRootRefiner(config, solver=_solver)
    )

    polynomial_cache = _PolynomialCache(lines)
    records: list[CandidateRecord] = []
    solves: dict[tuple[CandidateFamily, tuple[int, ...], int], SystemSolveRecord] = {}
    failures: dict[tuple[CandidateFamily, tuple[int, ...], int, str, str], SolverFailure] = {}
    triple_systems = stationary_roots = 0
    quadruple_systems = vertex_roots = 0

    def register_solve(
        family: CandidateFamily,
        indices: tuple[int, ...],
        precision: int,
        roots: Sequence[RealRoot],
    ) -> None:
        solves[(family, tuple(indices), precision)] = SystemSolveRecord(
            family, tuple(indices), precision, tuple(roots)
        )

    def register_failure(
        family: CandidateFamily,
        indices: tuple[int, ...],
        precision: int,
        error: Exception,
    ) -> None:
        item = _failure(family, tuple(indices), precision, error)
        failures[(family, tuple(indices), precision, item.error_type, item.message)] = item

    # Stage 1: stationary points of every three-line trisector.
    for indices in combinations(range(len(lines)), 3):
        triple_systems += 1
        system = polynomial_cache.stationary_system(indices)
        try:
            roots = refiner.initial_roots(system.equations)
            register_solve("stationary", indices, config.initial_precision_bits, roots)
        except MsolveError as error:
            register_failure("stationary", indices, config.initial_precision_bits, error)
            continue
        stationary_roots += len(roots)
        for root_index in range(len(roots)):
            records.append(
                _certify_root(
                    family="stationary",
                    defining_lines=indices,
                    polynomials=system.equations,
                    original_roots=roots,
                    root_index=root_index,
                    refiner=refiner,
                    decision_function=lambda root, system=system, indices=indices: _stationary_decision(
                        root,
                        indices,
                        system,
                        polynomial_cache.line_count,
                        polynomial_cache.distance_difference,
                    ),
                    register_solve=register_solve,
                    register_failure=register_failure,
                )
            )

    # Stage 2: equidistant vertices of every four-line subset.  Determinants
    # may be shared across quadruples; eta is constructed once per quadruple.
    for indices in combinations(range(len(lines)), 4):
        quadruple_systems += 1
        equations = polynomial_cache.vertex_equations(indices)
        try:
            roots = refiner.initial_roots(equations)
            register_solve("vertex", indices, config.initial_precision_bits, roots)
        except MsolveError as error:
            register_failure("vertex", indices, config.initial_precision_bits, error)
            continue
        vertex_roots += len(roots)
        determinants = polynomial_cache.quadruple_determinants(indices)
        eta = _eta_polynomial(determinants)
        for root_index in range(len(roots)):
            records.append(
                _certify_root(
                    family="vertex",
                    defining_lines=indices,
                    polynomials=equations,
                    original_roots=roots,
                    root_index=root_index,
                    refiner=refiner,
                    decision_function=lambda root, indices=indices, determinants=determinants, eta=eta: _vertex_decision(
                        root,
                        indices,
                        determinants,
                        eta,
                        polynomial_cache.line_count,
                        polynomial_cache.distance_difference,
                    ),
                    register_solve=register_solve,
                    register_failure=register_failure,
                )
            )

    classifications = [
        record.classification for record in records if record.status == "accepted"
    ]
    final_deletions = tuple(
        record
        for record in records
        if record.status == "accepted" and record.final_deletion
    )
    if len(final_deletions) > 1:
        raise FinalDeletionMultiplicityError(final_deletions)
    diagnostics = CountDiagnostics(
        triple_systems,
        quadruple_systems,
        stationary_roots,
        vertex_roots,
        tuple(records),
        tuple(solves.values()),
        tuple(failures.values()),
        refiner.solver_calls,
    )
    return EventCountResult(
        classifications.count("deletion"),
        classifications.count("swap"),
        classifications.count("local_minimum"),
        classifications.count("local_maximum"),
        diagnostics,
        bool(final_deletions),
    )


# Public entry point; retain the versioned function name for compatibility.
count_events = count_events_v2


__all__ = [
    "CandidateRecord",
    "CertificationRound",
    "CertifiedSignCheck",
    "CountDiagnostics",
    "EventCountResult",
    "FinalDeletionMultiplicityError",
    "SolverFailure",
    "SystemSolveRecord",
    "count_events",
    "count_events_v2",
]
