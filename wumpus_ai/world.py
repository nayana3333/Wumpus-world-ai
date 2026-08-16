"""
Wumpus World environment (AIMA chapter 7 style).

The environment is *fully separated* from the agent: an agent never touches
this object's attributes, it only ever receives a `Percept`. That separation is
what makes the agents in `agents.py` genuine knowledge-based agents rather than
scripts that peek at the answer.

Grid convention
---------------
Cells are addressed as ``(row, col)`` with ``row = 0`` at the TOP of the screen.
The classic start square (bottom-left) is therefore ``(size - 1, 0)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]

# Screen-space directions: "up" decreases the row index.
DIRECTIONS: Dict[str, Coord] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Classic AIMA scoring.
REWARD_GOLD_ESCAPE = 1000
PENALTY_DEATH = -1000
PENALTY_ARROW = -10
PENALTY_ACTION = -1


def in_bounds(pos: Coord, size: int) -> bool:
    r, c = pos
    return 0 <= r < size and 0 <= c < size


def neighbors(pos: Coord, size: int) -> List[Coord]:
    """The four orthogonally adjacent cells that lie inside the cave."""
    r, c = pos
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [p for p in candidates if in_bounds(p, size)]


@dataclass
class Percept:
    """What the agent's five senses report in the current square."""

    stench: bool = False
    breeze: bool = False
    glitter: bool = False
    bump: bool = False
    scream: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass
class WumpusWorld:
    """A single episode of the Wumpus World."""

    size: int = 4
    pits_count: int = 3
    seed: Optional[int] = None

    # --- hidden state (agents must never read these) ---------------------
    pits: Set[Coord] = field(default_factory=set)
    wumpus: Coord = (0, 0)
    gold: Coord = (0, 0)

    # --- observable / bookkeeping state ----------------------------------
    agent: Coord = (0, 0)
    start: Coord = (0, 0)
    has_gold: bool = False
    has_arrow: bool = True
    wumpus_alive: bool = True
    scream_pending: bool = False
    bump_pending: bool = False
    game_over: bool = False
    won: bool = False
    outcome: str = ""
    status: str = "Welcome to the cave."
    score: int = 0
    steps: int = 0
    visited: Set[Coord] = field(default_factory=set)
    path: List[Coord] = field(default_factory=list)
    arrow_flight: List[Coord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: Optional[int] = None) -> None:
        """Lay out a fresh cave. A `seed` makes the layout reproducible."""
        self.seed = seed
        rng = random.Random(seed)

        self.size = max(3, min(12, int(self.size)))
        self.start = (self.size - 1, 0)
        self.agent = self.start

        # Only the start square is protected. Keeping its neighbours hazardous
        # is deliberate: "breeze on move one" is exactly the situation that
        # separates a reasoning agent from a lucky one.
        forbidden = {self.start}
        cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in forbidden
        ]

        max_pits = max(0, len(cells) - 2)
        self.pits_count = max(0, min(int(self.pits_count), max_pits))

        rng.shuffle(cells)
        self.pits = set(cells[: self.pits_count])
        remaining = [c for c in cells if c not in self.pits]

        self.wumpus = rng.choice(remaining)
        gold_options = [c for c in remaining if c != self.wumpus] or remaining
        self.gold = rng.choice(gold_options)

        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.scream_pending = False
        self.bump_pending = False
        self.game_over = False
        self.won = False
        self.outcome = ""
        self.status = "You are in the cave. You can only sense the square you are standing in."
        self.score = 0
        self.steps = 0
        self.visited = {self.agent}
        self.path = [self.agent]
        self.arrow_flight = []

    @property
    def pit_prior(self) -> float:
        """Prior probability that an arbitrary unknown square holds a pit."""
        candidates = self.size * self.size - 1
        return self.pits_count / candidates if candidates else 0.0

    # -------------------------------------------------------------- percepts
    def percept(self) -> Percept:
        """The percept vector for the square the agent currently occupies.

        Pure: calling it twice gives the same answer. The momentary percepts
        (bump, scream) are cleared at the *start* of the next action instead,
        so a plain state refresh never swallows them.
        """
        adj = neighbors(self.agent, self.size)
        return Percept(
            stench=self.wumpus_alive and self.wumpus in adj,
            breeze=any(a in self.pits for a in adj),
            glitter=(self.agent == self.gold and not self.has_gold),
            bump=self.bump_pending,
            scream=self.scream_pending,
        )

    # --------------------------------------------------------------- actions
    def act(self, action: str, direction: Optional[str] = None) -> Dict:
        """Apply one action. Returns the full serialisable state."""
        if self.game_over:
            return self.state()

        # Momentary percepts only survive until the next action is taken.
        self.bump_pending = False
        self.scream_pending = False
        self.arrow_flight = []

        action = (action or "").lower()
        if action == "move":
            self._move(direction)
        elif action == "grab":
            self._grab()
        elif action == "shoot":
            self._shoot(direction)
        elif action == "climb":
            self._climb()
        else:
            self.status = f"Unknown action: {action!r}"
        return self.state()

    def _charge(self, amount: int) -> None:
        self.score += amount

    def _move(self, direction: Optional[str]) -> None:
        delta = DIRECTIONS.get(direction or "")
        if delta is None:
            self.status = "That is not a direction."
            return

        self.steps += 1
        self._charge(PENALTY_ACTION)
        r, c = self.agent
        target = (r + delta[0], c + delta[1])

        if not in_bounds(target, self.size):
            self.bump_pending = True
            self.status = "Bump. There is a wall that way."
            return

        self.agent = target
        self.visited.add(target)
        self.path.append(target)
        self.status = f"Moved {direction} to {target}."

        if target in self.pits:
            self._die("pit", "You fell into a pit.")
        elif target == self.wumpus and self.wumpus_alive:
            self._die("wumpus", "The Wumpus got you.")

    def _grab(self) -> None:
        self.steps += 1
        self._charge(PENALTY_ACTION)
        if self.agent == self.gold and not self.has_gold:
            self.has_gold = True
            self.status = "Got the gold. Now get back to the start square."
        else:
            self.status = "Nothing to grab here."

    def _shoot(self, direction: Optional[str]) -> None:
        self.steps += 1
        self._charge(PENALTY_ACTION)
        if not self.has_arrow:
            self.status = "No arrows left."
            return
        delta = DIRECTIONS.get(direction or "")
        if delta is None:
            self.status = "You need a direction to shoot in."
            return

        self.has_arrow = False
        self._charge(PENALTY_ARROW)

        r, c = self.agent
        flight: List[Coord] = []
        hit = False
        while True:
            r, c = r + delta[0], c + delta[1]
            if not in_bounds((r, c), self.size):
                break
            flight.append((r, c))
            if (r, c) == self.wumpus and self.wumpus_alive:
                self.wumpus_alive = False
                self.scream_pending = True
                hit = True
                break

        self.arrow_flight = flight
        if hit:
            self.status = "You hear a scream. The Wumpus is dead."
        else:
            self.status = "The arrow hit a wall."

    def _climb(self) -> None:
        self.steps += 1
        self._charge(PENALTY_ACTION)
        if self.agent != self.start:
            self.status = "You can only climb out from the start square."
            return

        self.game_over = True
        if self.has_gold:
            self._charge(REWARD_GOLD_ESCAPE)
            self.won = True
            self.outcome = "win"
            self.status = "You climbed out with the gold. You win."
        else:
            self.outcome = "escaped"
            self.status = "You climbed out without the gold."

    def _die(self, cause: str, message: str) -> None:
        self._charge(PENALTY_DEATH)
        self.game_over = True
        self.won = False
        self.outcome = cause
        self.status = message

    # ----------------------------------------------------------------- state
    def state(self, reveal: bool = False) -> Dict:
        """Serialise the world for the browser.

        Hidden cells are only exposed when `reveal` is set or the episode has
        finished - the UI cannot accidentally leak the solution.
        """
        show_all = reveal or self.game_over
        p = self.percept()

        tiles = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                seen = pos in self.visited or show_all
                row.append(
                    {
                        "r": r,
                        "c": c,
                        "visited": pos in self.visited,
                        "seen": seen,
                        "pit": (pos in self.pits) if seen else None,
                        "wumpus": (pos == self.wumpus) if seen else None,
                        "gold": (pos == self.gold and not self.has_gold) if seen else None,
                        "start": pos == self.start,
                    }
                )
            tiles.append(row)

        return {
            "size": self.size,
            "pits_count": self.pits_count,
            "pit_prior": round(self.pit_prior, 4),
            "seed": self.seed,
            "tiles": tiles,
            "agent": list(self.agent),
            "start": list(self.start),
            "path": [list(p_) for p_ in self.path],
            "has_gold": self.has_gold,
            "has_arrow": self.has_arrow,
            "wumpus_alive": self.wumpus_alive,
            "game_over": self.game_over,
            "won": self.won,
            "outcome": self.outcome,
            "status": self.status,
            "score": self.score,
            "steps": self.steps,
            "percept": p.as_dict(),
            "arrow_flight": [list(a) for a in self.arrow_flight],
            "revealed": show_all,
        }
