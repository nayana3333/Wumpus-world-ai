"""
Knowledge base for the Wumpus World.

This module is the "AI" half of the project. It is told nothing but percepts;
everything else is *derived*. Two inference mechanisms run side by side:

1. Logical entailment (sound, exact)
   - A visited square is definitely pit-free and Wumpus-free (you survived it).
   - No breeze at X  =>  no pit in any neighbour of X.
   - No stench at X  =>  no Wumpus in any neighbour of X.
   - Because there is exactly one Wumpus, "stench at X <=> Wumpus adjacent to
     X" is a biconditional, so the Wumpus can often be pinned down uniquely.

2. Probabilistic inference by model counting (AIMA §13.7)
   When logic is silent - every unvisited square *could* hold a pit - the agent
   still has to move. We enumerate every assignment of pits to the frontier
   (unknown squares adjacent to something we have visited) that is consistent
   with every breeze percept, weight each model by the pit prior, and read off
   the marginal P(pit | percepts) for each square.

   Exhaustive enumeration is 2^|frontier|, which is only tractable because the
   frontier is split into independent components first: two frontier squares
   interact only if some visited square is adjacent to both. In practice this
   keeps each component well under 20 variables even on a 10x10 cave.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .logic import MAX_PROVER_SIZE, CnfKnowledgeBase
from .world import Coord, neighbors

# Above this many variables in a single component we stop enumerating and
# switch to weighted sampling; 2^20 models is roughly the practical ceiling.
EXACT_ENUMERATION_LIMIT = 20
SAMPLE_COUNT = 20000


class KnowledgeBase:
    """Everything the agent has worked out about the cave so far."""

    def __init__(self, size: int, pit_prior: float, start: Optional[Coord] = None) -> None:
        self.size = size
        self.pit_prior = min(max(pit_prior, 0.001), 0.9)
        self.start = start or (size - 1, 0)
        self.visited: Set[Coord] = set()
        self.percepts: Dict[Coord, Dict[str, bool]] = {}
        self.wumpus_dead = False
        self.derivations: List[Dict[str, str]] = []
        self._seen_derivations: Set[str] = set()
        self._cache: Optional[Dict] = None

        # The CNF twin. Model counting already answers "is this square safe"
        # exactly, so the prover is not in the hot path; it is here so the
        # safety claims can be re-derived by resolution and shown as a proof.
        # Building "at most one Wumpus" is O(n^4) in the board side, so big
        # caves skip it.
        self.cnf: Optional[CnfKnowledgeBase] = None
        if size <= MAX_PROVER_SIZE:
            self.cnf = CnfKnowledgeBase(size)
            self.cnf.add_wumpus_axioms(self.start)

    # ------------------------------------------------------------------ TELL
    def tell(self, pos: Coord, percept: Dict[str, bool], wumpus_dead: bool = False) -> None:
        """Record a percept observed at `pos`. This is the only input channel."""
        pos = tuple(pos)  # type: ignore[assignment]
        self.visited.add(pos)
        stored = self.percepts.setdefault(pos, {"breeze": False, "stench": False, "glitter": False})
        stored["breeze"] = bool(percept.get("breeze"))
        stored["glitter"] = bool(percept.get("glitter"))
        # A stench recorded before the Wumpus died stays true of that moment,
        # but once it is dead stench stops being generated anywhere.
        if not self.wumpus_dead:
            stored["stench"] = bool(percept.get("stench"))
        if percept.get("scream"):
            self.wumpus_dead = True
        if wumpus_dead:
            self.wumpus_dead = True
        if self.cnf is not None:
            self.cnf.tell(pos, stored["breeze"], stored["stench"], scream=self.wumpus_dead)
        self._cache = None
        self._explain_local(pos, stored)

    # ------------------------------------------------------------- geography
    def all_cells(self) -> List[Coord]:
        return [(r, c) for r in range(self.size) for c in range(self.size)]

    def unknown_cells(self) -> List[Coord]:
        return [cell for cell in self.all_cells() if cell not in self.visited]

    def frontier(self) -> List[Coord]:
        """Unknown squares adjacent to at least one visited square."""
        return [
            cell
            for cell in self.unknown_cells()
            if any(n in self.visited for n in neighbors(cell, self.size))
        ]

    # ------------------------------------------------------- logical entailment
    def definitely_no_pit(self) -> Set[Coord]:
        """Squares proved pit-free: visited, or adjacent to a breeze-free square."""
        proved = set(self.visited)
        for pos, p in self.percepts.items():
            if not p["breeze"]:
                proved.update(neighbors(pos, self.size))
        return proved

    def definitely_no_wumpus(self) -> Set[Coord]:
        if self.wumpus_dead:
            return set(self.all_cells())
        proved = set(self.visited)
        for pos, p in self.percepts.items():
            if not p["stench"]:
                proved.update(neighbors(pos, self.size))
        return proved

    def wumpus_candidates(self) -> Set[Coord]:
        """Squares where the Wumpus could still be, given every stench percept.

        Uses the biconditional: with exactly one Wumpus, a square X smells iff
        the Wumpus sits in a neighbour of X.
        """
        if self.wumpus_dead:
            return set()
        candidates = set()
        for cell in self.unknown_cells():
            consistent = True
            for v, p in self.percepts.items():
                should_smell = cell in neighbors(v, self.size)
                if p["stench"] != should_smell:
                    consistent = False
                    break
            if consistent:
                candidates.add(cell)
        if not candidates:
            # Should be unreachable, but never let a contradiction crash a game.
            candidates = set(self.unknown_cells())
        return candidates

    # -------------------------------------------------- probabilistic inference
    def pit_probabilities(self) -> Dict[Coord, float]:
        """P(pit in cell | all breeze percepts), by weighted model counting."""
        probs: Dict[Coord, float] = {}
        for cell in self.all_cells():
            probs[cell] = 0.0

        proved_clear = self.definitely_no_pit()
        frontier = [c for c in self.frontier() if c not in proved_clear]
        frontier_set = set(frontier)

        # Constraints: one per visited square, over its unknown neighbours.
        constraints: List[Tuple[bool, Tuple[Coord, ...]]] = []
        for pos, p in self.percepts.items():
            unknown_nb = tuple(n for n in neighbors(pos, self.size) if n in frontier_set)
            if unknown_nb:
                constraints.append((p["breeze"], unknown_nb))

        components = self._split_components(frontier, constraints)

        for cells, cons in components:
            marginals = self._component_marginals(cells, cons)
            probs.update(marginals)

        # Squares we have never been near are governed purely by the prior.
        for cell in self.unknown_cells():
            if cell not in frontier_set and cell not in proved_clear:
                probs[cell] = self.pit_prior

        for cell in proved_clear:
            probs[cell] = 0.0
        return probs

    def _split_components(
        self,
        frontier: Sequence[Coord],
        constraints: Sequence[Tuple[bool, Tuple[Coord, ...]]],
    ) -> List[Tuple[List[Coord], List[Tuple[bool, Tuple[Coord, ...]]]]]:
        """Group frontier squares that share a constraint; solve groups apart."""
        parent: Dict[Coord, Coord] = {cell: cell for cell in frontier}

        def find(x: Coord) -> Coord:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: Coord, b: Coord) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for _, cells in constraints:
            for other in cells[1:]:
                union(cells[0], other)

        groups: Dict[Coord, List[Coord]] = {}
        for cell in frontier:
            groups.setdefault(find(cell), []).append(cell)

        grouped_constraints: Dict[Coord, List[Tuple[bool, Tuple[Coord, ...]]]] = {
            root: [] for root in groups
        }
        for constraint in constraints:
            root = find(constraint[1][0])
            grouped_constraints[root].append(constraint)

        return [(cells, grouped_constraints[root]) for root, cells in groups.items()]

    def _component_marginals(
        self,
        cells: List[Coord],
        constraints: List[Tuple[bool, Tuple[Coord, ...]]],
    ) -> Dict[Coord, float]:
        """Exact marginals inside one independent component."""
        n = len(cells)
        index = {cell: i for i, cell in enumerate(cells)}
        prior = self.pit_prior

        if n > EXACT_ENUMERATION_LIMIT:
            return self._sampled_marginals(cells, constraints)

        totals = [0.0] * n
        total_weight = 0.0

        compiled = [
            (breeze, tuple(index[c] for c in cs)) for breeze, cs in constraints
        ]

        for bits in itertools.product((0, 1), repeat=n):
            ok = True
            for breeze, idxs in compiled:
                has_pit = any(bits[i] for i in idxs)
                if has_pit != breeze:
                    ok = False
                    break
            if not ok:
                continue
            weight = 1.0
            for b in bits:
                weight *= prior if b else (1.0 - prior)
            total_weight += weight
            for i, b in enumerate(bits):
                if b:
                    totals[i] += weight

        if total_weight <= 0:
            # Contradictory percepts (impossible in a well-formed game): back
            # off to the prior rather than dividing by zero.
            return {cell: prior for cell in cells}
        return {cell: totals[i] / total_weight for i, cell in enumerate(cells)}

    def _sampled_marginals(
        self,
        cells: List[Coord],
        constraints: List[Tuple[bool, Tuple[Coord, ...]]],
    ) -> Dict[Coord, float]:
        """Rejection sampling fallback for very large frontier components."""
        rng = random.Random(1234)
        index = {cell: i for i, cell in enumerate(cells)}
        compiled = [(b, tuple(index[c] for c in cs)) for b, cs in constraints]
        prior = self.pit_prior
        n = len(cells)
        counts = [0] * n
        accepted = 0
        for _ in range(SAMPLE_COUNT):
            bits = [1 if rng.random() < prior else 0 for _ in range(n)]
            if all(any(bits[i] for i in idxs) == breeze for breeze, idxs in compiled):
                accepted += 1
                for i, b in enumerate(bits):
                    counts[i] += b
        if accepted == 0:
            return {cell: prior for cell in cells}
        return {cell: counts[i] / accepted for i, cell in enumerate(cells)}

    # ------------------------------------------------------------- assessment
    def assess(self) -> Dict:
        """Full risk picture for every square. Cached until the next TELL."""
        if self._cache is not None:
            return self._cache

        pit_probs = self.pit_probabilities()
        candidates = self.wumpus_candidates()
        wumpus_prob: Dict[Coord, float] = {}
        share = 1.0 / len(candidates) if candidates else 0.0
        for cell in self.all_cells():
            wumpus_prob[cell] = share if cell in candidates else 0.0

        cells: Dict[str, Dict] = {}
        safe: Set[Coord] = set()
        deadly: Set[Coord] = set()
        frontier_set = set(self.frontier())

        for cell in self.all_cells():
            pp = pit_probs.get(cell, 0.0)
            wp = wumpus_prob.get(cell, 0.0)
            # Independent-hazard approximation: the two dangers are separate
            # objects, so survival probability multiplies.
            danger = 1.0 - (1.0 - pp) * (1.0 - wp)
            if cell in self.visited:
                pp = wp = danger = 0.0

            if danger <= 1e-9:
                label = "safe"
                safe.add(cell)
            elif danger >= 1.0 - 1e-9:
                label = "deadly"
                deadly.add(cell)
            elif danger < 0.2:
                label = "likely-safe"
            elif danger < 0.5:
                label = "risky"
            else:
                label = "dangerous"

            cells[f"{cell[0]},{cell[1]}"] = {
                "pit": round(pp, 4),
                "wumpus": round(wp, 4),
                "danger": round(danger, 4),
                "label": label,
                "visited": cell in self.visited,
                "frontier": cell in frontier_set,
            }

        self._explain_global(candidates, pit_probs, safe, deadly)

        self._cache = {
            "cells": cells,
            "safe": [list(c) for c in sorted(safe)],
            "deadly": [list(c) for c in sorted(deadly)],
            "wumpus_candidates": [list(c) for c in sorted(candidates)],
            "wumpus_dead": self.wumpus_dead,
            "frontier": [list(c) for c in sorted(frontier_set)],
            "visited": [list(c) for c in sorted(self.visited)],
            "derivations": self.derivations[-40:],
            "pit_prior": round(self.pit_prior, 4),
            # What we smelled in each square we have stood in - the UI paints
            # these back onto the board as a memory aid.
            "percept_memory": {
                f"{r},{c}": {"breeze": p["breeze"], "stench": p["stench"]}
                for (r, c), p in self.percepts.items()
            },
        }
        return self._cache

    def prove(self, cell: Coord) -> Dict:
        """Ask the CNF side what it can prove about one square, with the proof.

        Model counting and resolution are two routes to the same conclusion;
        `test_prover_agrees_with_model_counting` checks they never disagree.
        """
        if self.cnf is None:
            return {
                "available": False,
                "reason": f"The prover is limited to boards up to {MAX_PROVER_SIZE}x{MAX_PROVER_SIZE}.",
            }

        cnf = self.cnf
        label = f"({cell[0]},{cell[1]})"
        questions = (
            ("no_pit", -cnf.pit(cell), f"no pit in {label}"),
            ("pit", cnf.pit(cell), f"a pit in {label}"),
            ("no_wumpus", -cnf.wumpus(cell), f"no Wumpus in {label}"),
            ("wumpus", cnf.wumpus(cell), f"the Wumpus in {label}"),
        )

        # Each entails() is a DPLL solve, so run each question once and derive
        # the verdict from the answers instead of calling proves_safe /
        # proves_deadly, which would solve the same four all over again.
        answers = {name: cnf.entails(lit) for name, lit, _ in questions}
        claims = [
            {"claim": text, "literal": cnf.name(lit), "steps": cnf.proof(lit) or []}
            for name, lit, text in questions
            if answers[name]
        ]

        safe = answers["no_pit"] and (cnf.wumpus_dead or answers["no_wumpus"])
        deadly = answers["pit"] or (not cnf.wumpus_dead and answers["wumpus"])

        return {
            "available": True,
            "cell": list(cell),
            "clauses": len(cnf.clauses),
            "entailed": claims,
            "assumptions": cnf.assumption_lines()[-14:],
            "verdict": (
                "provably safe" if safe
                else "provably deadly" if deadly
                else "undetermined by logic alone"
            ),
        }

    # ----------------------------------------------------------- explanations
    def _add_derivation(self, rule: str, text: str) -> None:
        key = f"{rule}|{text}"
        if key in self._seen_derivations:
            return
        self._seen_derivations.add(key)
        self.derivations.append({"rule": rule, "text": text})

    def _explain_local(self, pos: Coord, p: Dict[str, bool]) -> None:
        """Narrate the immediate, purely logical conclusions from one percept."""
        label = f"({pos[0]},{pos[1]})"
        nbs = [n for n in neighbors(pos, self.size) if n not in self.visited]
        nb_text = ", ".join(f"({r},{c})" for r, c in nbs) or "-"

        if not p["breeze"] and not p["stench"]:
            if nbs:
                self._add_derivation(
                    "R1",
                    f"No breeze and no stench at {label} ⟹ {nb_text} are provably SAFE.",
                )
        else:
            if p["breeze"]:
                self._add_derivation(
                    "R2",
                    f"Breeze at {label} ⟹ at least one of {nb_text} contains a pit.",
                )
            else:
                self._add_derivation(
                    "R3", f"No breeze at {label} ⟹ no pit in {nb_text}."
                )
            if p["stench"]:
                self._add_derivation(
                    "R4",
                    f"Stench at {label} ⟹ the Wumpus is in one of {nb_text}.",
                )
            else:
                self._add_derivation(
                    "R5", f"No stench at {label} ⟹ no Wumpus in {nb_text}."
                )
        if p["glitter"]:
            self._add_derivation("R6", f"Glitter at {label} ⟹ the gold is HERE. Grab it.")

    def _explain_global(
        self,
        candidates: Set[Coord],
        pit_probs: Dict[Coord, float],
        safe: Set[Coord],
        deadly: Set[Coord],
    ) -> None:
        if self.wumpus_dead:
            self._add_derivation("W0", "Scream heard ⟹ the Wumpus is dead; stench is no longer a threat.")
        elif len(candidates) == 1:
            (cell,) = tuple(candidates)
            self._add_derivation(
                "W1",
                f"Resolution over every stench percept leaves exactly one model ⟹ "
                f"the Wumpus is at ({cell[0]},{cell[1]}). It is now a shootable target.",
            )
        elif 1 < len(candidates) <= 3:
            listed = ", ".join(f"({r},{c})" for r, c in sorted(candidates))
            self._add_derivation(
                "W2", f"Wumpus narrowed to {len(candidates)} candidates: {listed}."
            )

        for cell in sorted(deadly):
            if cell in self.visited:
                continue
            if pit_probs.get(cell, 0.0) >= 1.0 - 1e-9:
                self._add_derivation(
                    "P1",
                    f"Every model consistent with the breeze percepts puts a pit at "
                    f"({cell[0]},{cell[1]}) ⟹ P(pit) = 1. Never enter.",
                )
