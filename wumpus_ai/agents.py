"""
The five agents. They share one interface::

    action = agent.decide(kb, observation)
    # -> {"action": "move"|"grab"|"shoot"|"climb", "direction": str|None,
    #     "reason": str, "confidence": float}

An agent gets the `KnowledgeBase` (built from percepts) and an `observation`
describing its own body: position, whether it is carrying gold, whether the
arrow is spent. None of them can see the map.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .kb import KnowledgeBase
from .world import Coord, DIRECTIONS, neighbors

Observation = Dict


def direction_between(a: Coord, b: Coord) -> Optional[str]:
    """The move that takes you from adjacent square `a` to square `b`."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    for name, delta in DIRECTIONS.items():
        if delta == (dr, dc):
            return name
    return None


def bfs_path(start: Coord, goals: Set[Coord], passable: Set[Coord], size: int) -> List[Coord]:
    """Shortest route from `start` to the nearest goal, walking only on
    `passable` squares. Returns [] when no such route exists."""
    if start in goals:
        return [start]
    frontier = deque([start])
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    while frontier:
        current = frontier.popleft()
        for nxt in neighbors(current, size):
            if nxt in came_from:
                continue
            if nxt not in passable and nxt not in goals:
                continue
            came_from[nxt] = current
            if nxt in goals:
                path = [nxt]
                while came_from[path[-1]] is not None:
                    path.append(came_from[path[-1]])  # type: ignore[arg-type]
                return list(reversed(path))
            frontier.append(nxt)
    return []


class BaseAgent:
    name = "base"
    label = "Base"
    blurb = ""

    def __init__(self, **kwargs) -> None:
        self.reset()

    def reset(self) -> None:
        """Called at the start of each episode. Override to clear per-episode state."""

    def decide(self, kb: KnowledgeBase, obs: Observation) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _act(action: str, direction: Optional[str], reason: str, confidence: float = 1.0) -> Dict:
        return {
            "action": action,
            "direction": direction,
            "reason": reason,
            "confidence": round(confidence, 3),
        }


class RandomAgent(BaseAgent):
    """The control group. Moves at random; grabs gold if it trips over it."""

    name = "random"
    label = "Random Walker"
    blurb = "No memory, no inference. The baseline every other agent must beat."

    def __init__(self, seed: Optional[int] = None, **kwargs) -> None:
        self.rng = random.Random(seed)
        super().__init__(**kwargs)

    def decide(self, kb: KnowledgeBase, obs: Observation) -> Dict:
        pos = tuple(obs["pos"])
        if obs["percept"].get("glitter") and not obs["has_gold"]:
            return self._act("grab", None, "Glitter here - grab it.", 1.0)
        if obs["has_gold"] and pos == tuple(obs["start"]):
            return self._act("climb", None, "Carrying gold and standing on the exit.", 1.0)
        options = [d for d, (dr, dc) in DIRECTIONS.items()
                   if 0 <= pos[0] + dr < kb.size and 0 <= pos[1] + dc < kb.size]
        direction = self.rng.choice(options)
        return self._act("move", direction, f"Random choice: {direction}.", 0.25)


class ReflexAgent(BaseAgent):
    """Percept-driven reflex rules only - no model of the world beyond 'visited'.

    This is roughly the agent the original project shipped, kept as a
    comparison point. It applies the one-step safety rule (a square next to a
    breeze-free, stench-free square is safe) but never reasons across percepts,
    so it stalls or gambles as soon as the cave gets interesting.
    """

    name = "reflex"
    label = "Reflex Agent"
    blurb = "Single-rule safety check, prefers unvisited neighbours. Simple and fragile."

    def __init__(self, seed: Optional[int] = None, **kwargs) -> None:
        self.rng = random.Random(seed)
        super().__init__(**kwargs)

    def decide(self, kb: KnowledgeBase, obs: Observation) -> Dict:
        pos = tuple(obs["pos"])
        start = tuple(obs["start"])
        percept = obs["percept"]

        if percept.get("glitter") and not obs["has_gold"]:
            return self._act("grab", None, "Glitter percept ⟹ gold is in this square.", 1.0)
        if obs["has_gold"] and pos == start:
            return self._act("climb", None, "Gold in hand and standing on the exit.", 1.0)

        locally_safe = kb.definitely_no_pit() & kb.definitely_no_wumpus()
        nbs = neighbors(pos, kb.size)
        unvisited_safe = [n for n in nbs if n in locally_safe and n not in kb.visited]
        if unvisited_safe:
            target = self.rng.choice(unvisited_safe)
            return self._act("move", direction_between(pos, target),
                             f"({target[0]},{target[1]}) is one-step safe and unexplored.", 0.9)

        visited_nbs = [n for n in nbs if n in kb.visited]
        if visited_nbs:
            target = self.rng.choice(visited_nbs)
            return self._act("move", direction_between(pos, target),
                             "No safe frontier next to me - backtracking.", 0.5)
        target = self.rng.choice(nbs)
        return self._act("move", direction_between(pos, target), "Out of ideas - guessing.", 0.2)


class LogicAgent(BaseAgent):
    """Acts only on squares the knowledge base proves are safe.

    Decision procedure, in priority order:
      1. Glitter here            -> grab.
      2. Carrying gold           -> plan a route home through proven-safe squares, then climb.
      3. Proven-safe unexplored  -> plan a route to the nearest one.
      4. Wumpus pinned down and arrow in hand -> line up and shoot to unlock territory.
      5. Nothing safe left       -> gamble on the least dangerous frontier square,
                                    but only if its risk is under `risk_tolerance`.
      6. Otherwise               -> go home and climb out with whatever score is left.

    Routing between distant squares is BFS over the proven-safe set, so it
    retreats along known corridors rather than across unexplored ground.
    """

    name = "logic"
    label = "Logic Agent"
    blurb = "Propositional entailment only. Moves when it can prove safety, retreats when it cannot."
    risk_tolerance = 0.0

    def __init__(self, risk_tolerance: Optional[float] = None, seed: Optional[int] = None, **kwargs) -> None:
        if risk_tolerance is not None:
            self.risk_tolerance = float(risk_tolerance)
        self.rng = random.Random(seed)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ main
    def decide(self, kb: KnowledgeBase, obs: Observation) -> Dict:
        pos = tuple(obs["pos"])
        start = tuple(obs["start"])
        percept = obs["percept"]
        assessment = kb.assess()

        # 1. Gold in this square.
        if percept.get("glitter") and not obs["has_gold"]:
            return self._act("grab", None, "Glitter ⟹ the gold is in this square. Grab.", 1.0)

        safe = {tuple(c) for c in assessment["safe"]}
        safe |= kb.visited
        reachable_safe = safe

        # 2. Head home with the gold.
        if obs["has_gold"]:
            if pos == start:
                return self._act("climb", None, "Gold secured and standing on the exit - climb out.", 1.0)
            step = self._step_towards({start}, pos, reachable_safe, kb)
            if step:
                return self._act("move", step,
                                 "Carrying the gold: retracing a proven-safe route to the exit.", 1.0)
            step = self._greedy_step(pos, start, kb, assessment)
            if step:
                return self._act("move", step,
                                 "No fully safe route home - taking the least dangerous step.", 0.5)

        # 3. Nearest proven-safe unexplored square.
        unexplored_safe = {c for c in safe if c not in kb.visited}
        if unexplored_safe:
            step = self._step_towards(unexplored_safe, pos, reachable_safe | unexplored_safe, kb)
            if step:
                return self._act("move", step,
                                 "The KB entails at least one unexplored square is safe - going there.", 1.0)

        # 4. Use the arrow to open up the map.
        shot = self._consider_shooting(kb, obs, assessment)
        if shot:
            return shot

        # 5. Calculated risk.
        gamble = self._consider_gamble(kb, obs, assessment, reachable_safe)
        if gamble:
            return gamble

        # 6. Cut losses.
        if pos == start:
            return self._act("climb", None,
                             "Every remaining square is an unacceptable risk - climbing out alive.", 0.8)
        step = self._step_towards({start}, pos, reachable_safe, kb)
        if step:
            return self._act("move", step, "Nothing safe left to explore - walking back to the exit.", 0.8)

        step = self._greedy_step(pos, start, kb, assessment)
        if step:
            return self._act("move", step, "Trapped: forcing a path back toward the exit.", 0.3)
        return self._act("climb", None, "No legal move available.", 0.1)

    # ------------------------------------------------------------- behaviours
    def _consider_shooting(self, kb: KnowledgeBase, obs: Observation, assessment: Dict) -> Optional[Dict]:
        """Shoot when the Wumpus is uniquely located and standing in our way."""
        if not obs["has_arrow"] or kb.wumpus_dead:
            return None
        candidates = [tuple(c) for c in assessment["wumpus_candidates"]]
        if len(candidates) != 1:
            return None
        target = candidates[0]
        pos = tuple(obs["pos"])

        # Only worth an arrow if killing it actually frees up a square we want.
        if not self._wumpus_blocks_progress(kb, assessment, target):
            return None

        if pos[0] == target[0] or pos[1] == target[1]:
            direction = self._aim(pos, target)
            if direction:
                return self._act(
                    "shoot", direction,
                    f"Wumpus proven to be at ({target[0]},{target[1]}) and I am aligned with it - firing {direction}.",
                    1.0,
                )

        # Not aligned: walk to a safe square that shares a row or column.
        safe = {tuple(c) for c in assessment["safe"]} | kb.visited
        firing_positions = {
            c for c in safe
            if (c[0] == target[0] or c[1] == target[1]) and c != target
        }
        if firing_positions:
            step = self._step_towards(firing_positions, pos, safe, kb)
            if step:
                return self._act("move", step,
                                 f"Manoeuvring to a firing line on the Wumpus at ({target[0]},{target[1]}).", 0.9)
        return None

    def _wumpus_blocks_progress(self, kb: KnowledgeBase, assessment: Dict, target: Coord) -> bool:
        """True when killing the Wumpus would make some square newly safe."""
        info = assessment["cells"]
        for cell in kb.unknown_cells():
            if cell == target:
                continue
            data = info.get(f"{cell[0]},{cell[1]}", {})
            if data.get("pit", 1.0) <= 1e-9 and data.get("wumpus", 0.0) > 1e-9:
                return True
        # Or the Wumpus itself sits on a square with no pit risk - killing it
        # turns that square into a usable corridor.
        data = info.get(f"{target[0]},{target[1]}", {})
        return data.get("pit", 1.0) <= 1e-9

    def _consider_gamble(
        self, kb: KnowledgeBase, obs: Observation, assessment: Dict, safe: Set[Coord]
    ) -> Optional[Dict]:
        """Pick the least dangerous frontier square, if it clears our threshold."""
        if self.risk_tolerance <= 0:
            return None
        info = assessment["cells"]
        options: List[Tuple[float, Coord]] = []
        for cell in kb.frontier():
            data = info.get(f"{cell[0]},{cell[1]}")
            if not data or data["danger"] >= 1.0 - 1e-9:
                continue
            if data["danger"] > self.risk_tolerance:
                continue
            options.append((data["danger"], cell))
        if not options:
            return None

        options.sort()
        best_risk, best_cell = options[0]
        pos = tuple(obs["pos"])
        step = self._step_towards({best_cell}, pos, safe | {best_cell}, kb)
        if not step:
            return None
        pit = info[f"{best_cell[0]},{best_cell[1]}"]["pit"]
        wum = info[f"{best_cell[0]},{best_cell[1]}"]["wumpus"]
        return self._act(
            "move", step,
            f"No provably safe square left. Model counting puts ({best_cell[0]},{best_cell[1]}) at "
            f"P(pit)={pit:.0%}, P(Wumpus)={wum:.0%} - the lowest risk available, and under my "
            f"{self.risk_tolerance:.0%} tolerance.",
            1.0 - best_risk,
        )

    # ---------------------------------------------------------------- routing
    def _step_towards(self, goals: Set[Coord], pos: Coord, passable: Set[Coord], kb: KnowledgeBase) -> Optional[str]:
        path = bfs_path(pos, goals, passable, kb.size)
        if len(path) < 2:
            return None
        return direction_between(pos, path[1])

    def _greedy_step(self, pos: Coord, goal: Coord, kb: KnowledgeBase, assessment: Dict) -> Optional[str]:
        """Last resort: the adjacent square that is closest to `goal` and least deadly."""
        info = assessment["cells"]
        best: Optional[Tuple[float, int, str]] = None
        for nxt in neighbors(pos, kb.size):
            data = info.get(f"{nxt[0]},{nxt[1]}", {"danger": 1.0})
            direction = direction_between(pos, nxt)
            if not direction:
                continue
            key = (data["danger"], abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]), direction)
            if best is None or key < best:
                best = key
        return best[2] if best else None

    @staticmethod
    def _aim(pos: Coord, target: Coord) -> Optional[str]:
        if pos[0] == target[0]:
            return "right" if target[1] > pos[1] else "left"
        if pos[1] == target[1]:
            return "down" if target[0] > pos[0] else "up"
        return None


class ProbabilisticAgent(LogicAgent):
    """Logic agent that will take a calculated bet when logic runs out.

    Same inference, different policy. Pure logic often refuses to move and
    climbs out with a small positive score; this one accepts frontier squares
    below a risk threshold. It wins more often and dies more often.
    """

    name = "probabilistic"
    label = "Probabilistic Agent"
    blurb = "Same knowledge base, plus Bayesian model counting to price risk when nothing is provably safe."
    risk_tolerance = 0.34


class QLearningAgent(BaseAgent):
    """Tabular Q-learning. Learns from reward instead of reasoning from percepts.

    Everything above this class is told the rules of the cave. This one is not.
    It sees a compact state key and a scalar reward, and has to work out on its
    own that a breeze is bad news.

    State key (5184 states):
        breeze, stench, glitter, has_gold, standing on the exit,
        and whether each of the four neighbours is a wall / unvisited / visited.

    The arrow is left out of the action space on purpose. Adding four shoot
    actions nearly doubles it, and the arrow matters in well under 10% of caves,
    so it would cost far more in sample efficiency than it could win back.
    """

    name = "qlearning"
    label = "Q-Learning Agent"
    blurb = "Tabular reinforcement learning. Knows no rules; learns the cave from reward alone."

    ACTIONS = [
        ("move", "up"), ("move", "down"), ("move", "left"), ("move", "right"),
        ("grab", None), ("climb", None),
    ]

    def __init__(self, table: Optional[Dict] = None, epsilon: float = 0.0,
                 alpha: float = 0.15, gamma: float = 0.95, seed: Optional[int] = None, **kwargs) -> None:
        self.q: Dict[Tuple, List[float]] = table if table is not None else {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.trained_episodes = 0
        super().__init__(**kwargs)

    # ------------------------------------------------------------- state key
    def state_key(self, kb: KnowledgeBase, obs: Observation) -> Tuple:
        pos = tuple(obs["pos"])
        percept = obs["percept"]
        around = []
        for _, (dr, dc) in DIRECTIONS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < kb.size and 0 <= nxt[1] < kb.size):
                around.append(0)          # wall
            elif nxt in kb.visited:
                around.append(2)          # already been there
            else:
                around.append(1)          # unexplored
        return (
            bool(percept.get("breeze")),
            bool(percept.get("stench")),
            bool(percept.get("glitter")),
            bool(obs["has_gold"]),
            pos == tuple(obs["start"]),
            tuple(around),
        )

    def values(self, key: Tuple) -> List[float]:
        return self.q.setdefault(key, [0.0] * len(self.ACTIONS))

    # ---------------------------------------------------------------- acting
    def choose(self, key: Tuple) -> int:
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.ACTIONS))
        values = self.values(key)
        best = max(values)
        # Break ties randomly, otherwise an untrained table always picks "up".
        return self.rng.choice([i for i, v in enumerate(values) if v == best])

    def decide(self, kb: KnowledgeBase, obs: Observation) -> Dict:
        key = self.state_key(kb, obs)
        index = self.choose(key)
        action, direction = self.ACTIONS[index]
        values = self.values(key)
        spread = max(values) - min(values)
        confidence = 0.0 if spread == 0 else min(1.0, spread / 200.0)
        verb = f"{action}{' ' + direction if direction else ''}"
        return self._act(
            action, direction,
            f"Learned policy: Q={values[index]:.1f} for {verb} in this state "
            f"(trained on {self.trained_episodes} episodes).",
            confidence,
        )

    # -------------------------------------------------------------- learning
    def update(self, key: Tuple, index: int, reward: float, next_key: Optional[Tuple]) -> None:
        current = self.values(key)
        future = 0.0 if next_key is None else max(self.values(next_key))
        current[index] += self.alpha * (reward + self.gamma * future - current[index])


AGENT_REGISTRY = {
    cls.name: cls
    for cls in (RandomAgent, ReflexAgent, LogicAgent, ProbabilisticAgent, QLearningAgent)
}


def make_agent(name: str, **kwargs) -> BaseAgent:
    if name == "qlearning" and "table" not in kwargs:
        # Imported here, not at module scope: rl imports this module.
        from .rl import load

        trained = load()
        if trained is not None:
            trained.rng = random.Random(kwargs.get("seed"))
            return trained
    cls = AGENT_REGISTRY.get(name, ProbabilisticAgent)
    return cls(**kwargs)


def agent_catalogue() -> List[Dict]:
    return [
        {
            "name": cls.name,
            "label": cls.label,
            "blurb": cls.blurb,
            "risk_tolerance": getattr(cls, "risk_tolerance", None),
        }
        for cls in AGENT_REGISTRY.values()
    ]
