"""
Propositional logic layer: CNF knowledge base, DPLL, unit resolution.

Literals are signed integers (DIMACS style): +n means the symbol is true, -n
means it is false. Clauses are frozensets of literals.

Symbols:
    P(r,c)  there is a pit in (r,c)
    W(r,c)  the Wumpus is in (r,c)

Percepts are not symbols. A breeze at (r,c) is asserted directly as the clause
set for "P(n1) v P(n2) v ..." (or its negation), which keeps the symbol count
down to 2 * size^2.
"""

from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .world import Coord, neighbors

Clause = FrozenSet[int]

# DPLL over every square gets slow well before the UI does, and the marginal
# teaching value of a proof on a 12x12 board is nil. Above this the knowledge
# base falls back to model counting alone.
MAX_PROVER_SIZE = 8


class CnfKnowledgeBase:
    """A CNF clause store with DPLL entailment and unit-resolution proofs."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.clauses: List[Clause] = []
        self._names: Dict[int, str] = {}
        self._sources: Dict[Clause, str] = {}
        self.wumpus_dead = False
        self._built_wumpus_axioms = False
        # Clauses that are only true while the Wumpus is alive. A stench
        # recorded at time t says "the Wumpus is next door" -- a claim about
        # then, not about now. Leaving these in after the kill makes the KB
        # unsatisfiable, and an unsatisfiable KB entails everything, so DPLL
        # would cheerfully "prove" every square safe.
        self._alive_only: List[Clause] = []

    # ------------------------------------------------------------- symbols
    def _symbol(self, kind: str, cell: Coord, block: int) -> int:
        n = 1 + block * self.size * self.size + cell[0] * self.size + cell[1]
        self._names[n] = f"{kind}{cell[0]}{cell[1]}"
        return n

    def pit(self, cell: Coord) -> int:
        return self._symbol("P", cell, 0)

    def wumpus(self, cell: Coord) -> int:
        return self._symbol("W", cell, 1)

    def breeze(self, cell: Coord) -> int:
        return self._symbol("B", cell, 2)

    def stench(self, cell: Coord) -> int:
        return self._symbol("S", cell, 3)

    def name(self, lit: int) -> str:
        base = self._names.get(abs(lit), f"x{abs(lit)}")
        return base if lit > 0 else f"¬{base}"

    def show(self, clause: Clause) -> str:
        if not clause:
            return "⊥"
        return " ∨ ".join(sorted(self.name(l) for l in clause))

    # -------------------------------------------------------------- asserts
    def add(self, literals: Sequence[int], why: str = "", alive_only: bool = False) -> None:
        clause = frozenset(literals)
        if clause in self._sources:
            return
        self.clauses.append(clause)
        self._sources[clause] = why
        if alive_only:
            self._alive_only.append(clause)

    def cells(self) -> List[Coord]:
        return [(r, c) for r in range(self.size) for c in range(self.size)]

    def add_wumpus_axioms(self, start: Coord) -> None:
        """Exactly one Wumpus, and the entrance square is safe."""
        if self._built_wumpus_axioms:
            return
        self._built_wumpus_axioms = True
        cells = self.cells()
        self.add([-self.pit(start)], "the entrance square is never a pit")
        self.add([self.wumpus(c) for c in cells], "there is at least one Wumpus", alive_only=True)
        for a, b in itertools.combinations(cells, 2):
            self.add([-self.wumpus(a), -self.wumpus(b)], "there is at most one Wumpus")

    def _biconditional(self, sensor: int, causes: List[int], label: str) -> None:
        """sensor <-> (c1 v c2 v ...), written out in CNF.

        This is the textbook's R1-style sentence rather than a shortcut, and it
        matters for the proofs. Asserting "no breeze here, so no pit next door"
        directly would drop the conclusion straight into the KB and the Proof
        tab could only ever restate it. Keeping the rule and the observation
        separate means the conclusion has to actually be derived.
        """
        self.add([-sensor] + causes, label)
        for cause in causes:
            self.add([-cause, sensor], label)

    def tell(self, cell: Coord, breeze: bool, stench: bool, scream: bool = False) -> None:
        """Assert everything a percept at `cell` licenses."""
        nbs = neighbors(cell, self.size)
        self.add([-self.pit(cell)], f"stood in {cell} and survived")
        self.add([-self.wumpus(cell)], f"stood in {cell} and survived")

        self._biconditional(
            self.breeze(cell), [self.pit(n) for n in nbs],
            f"breeze at {cell} iff a pit is adjacent",
        )
        self.add(
            [self.breeze(cell) if breeze else -self.breeze(cell)],
            f"{'felt a' if breeze else 'no'} breeze at {cell}",
        )

        self._biconditional(
            self.stench(cell), [self.wumpus(n) for n in nbs],
            f"stench at {cell} iff the Wumpus is adjacent",
        )

        if scream or self.wumpus_dead:
            self.kill_wumpus()
        else:
            # Only the observation is retracted when the Wumpus dies. The rule
            # above stays and then forces "no stench" everywhere by itself.
            self.add(
                [self.stench(cell) if stench else -self.stench(cell)],
                f"{'smelled a' if stench else 'no'} stench at {cell}",
                alive_only=stench,
            )

    def kill_wumpus(self) -> None:
        if self.wumpus_dead:
            return
        self.wumpus_dead = True
        # Retract everything that was only true while it was alive, then assert
        # that every square is now Wumpus-free.
        retract = set(self._alive_only)
        self.clauses = [c for c in self.clauses if c not in retract]
        for clause in retract:
            self._sources.pop(clause, None)
        self._alive_only = []
        for c in self.cells():
            self.add([-self.wumpus(c)], "the Wumpus is dead")

    # ------------------------------------------------------------ inference
    def entails(self, literal: int) -> bool:
        """Does KB entail `literal`? True iff KB and its negation is unsat."""
        return not dpll_satisfiable(self.clauses + [frozenset([-literal])])

    def proves_safe(self, cell: Coord) -> bool:
        return self.entails(-self.pit(cell)) and (
            self.wumpus_dead or self.entails(-self.wumpus(cell))
        )

    def proves_deadly(self, cell: Coord) -> bool:
        return self.entails(self.pit(cell)) or (
            not self.wumpus_dead and self.entails(self.wumpus(cell))
        )

    # ---------------------------------------------------------------- proofs
    def proof(self, literal: int, limit: int = 400) -> Optional[List[Dict[str, str]]]:
        """A readable refutation of `not literal`, via unit resolution.

        Returns the resolution steps, or None when unit resolution alone is not
        enough (DPLL may still settle the question, it just is not a chain a
        human wants to read).
        """
        goal = frozenset([-literal])
        clauses = list(self.clauses) + [goal]
        # The goal is scratch for this query, not part of the knowledge base.
        # Writing it into self._sources would leak it into assumption_lines().
        sources = {goal: "assume the opposite"}

        derived: Dict[Clause, Tuple[Clause, Clause, int]] = {}
        known = set(clauses)
        units = [c for c in clauses if len(c) == 1]
        steps = 0

        while units and steps < limit:
            unit = units.pop(0)
            (lit,) = tuple(unit)
            for clause in list(known):
                if -lit not in clause:
                    continue
                resolvent = frozenset(clause - {-lit})
                if resolvent in known:
                    continue
                steps += 1
                known.add(resolvent)
                derived[resolvent] = (unit, clause, lit)
                if not resolvent:
                    return self._trace(resolvent, derived, sources)
                if len(resolvent) == 1:
                    units.append(resolvent)
        return None

    def _trace(self, empty: Clause, derived, sources: Dict[Clause, str]) -> List[Dict[str, str]]:
        """Walk the derivation graph back to the assumptions."""
        order: List[Clause] = []
        seen: Set[Clause] = set()

        def visit(clause: Clause) -> None:
            if clause in seen:
                return
            seen.add(clause)
            if clause in derived:
                a, b, _ = derived[clause]
                visit(a)
                visit(b)
                order.append(clause)

        visit(empty)

        def why(clause: Clause) -> str:
            return sources.get(clause) or self._sources.get(clause, "")

        out: List[Dict[str, str]] = []
        for clause in order:
            a, b, lit = derived[clause]
            out.append({
                "from": f"{self.show(a)}   and   {self.show(b)}",
                "on": self.name(lit),
                "gives": self.show(clause),
                "why": why(b) or why(a),
            })
        return out

    def assumption_lines(self) -> List[str]:
        """The asserted clauses, for showing alongside a proof."""
        return [
            f"{self.show(c)}    ({self._sources[c]})"
            for c in self.clauses
            if self._sources.get(c) and len(c) <= 4
        ]


def dpll_satisfiable(clauses: Sequence[Clause]) -> bool:
    """Standard DPLL: unit propagation, pure literals, then split.

    Only satisfiability is needed here (entailment is a refutation test), so no
    model is built up on the way down.
    """
    return _dpll([set(c) for c in clauses])


def _dpll(clauses: List[Set[int]]) -> bool:
    clauses, ok = _propagate_units(clauses)
    if not ok:
        return False
    if not clauses:
        return True

    # Pure literal: a symbol that only ever appears with one sign can be set
    # that way without ever falsifying anything.
    counts: Set[int] = set()
    for clause in clauses:
        counts |= clause
    pure = [l for l in counts if -l not in counts]
    if pure:
        chosen = set(pure)
        return _dpll([c for c in clauses if not (c & chosen)])

    # Split on a literal from the shortest clause.
    pivot = next(iter(min(clauses, key=len)))
    for value in (pivot, -pivot):
        branch = [set(c) for c in clauses if value not in c]
        for clause in branch:
            clause.discard(-value)
        if any(not c for c in branch):
            continue
        if _dpll(branch):
            return True
    return False


def _propagate_units(clauses: List[Set[int]]) -> Tuple[List[Set[int]], bool]:
    """Repeatedly satisfy unit clauses. False means a contradiction."""
    while True:
        unit = next((next(iter(c)) for c in clauses if len(c) == 1), None)
        if unit is None:
            return clauses, True
        reduced: List[Set[int]] = []
        for clause in clauses:
            if unit in clause:
                continue
            if -unit in clause:
                shorter = clause - {-unit}
                if not shorter:
                    return clauses, False
                reduced.append(shorter)
            else:
                reduced.append(set(clause))
        clauses = reduced
