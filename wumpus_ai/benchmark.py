"""
Batch evaluation harness.

Runs each agent over the *same* set of randomly generated caves (identical
seeds per episode) so the comparison is paired and fair, then reports win rate,
mean score, survival rate and average episode length. This is the table that
belongs in an AI project report.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

from .agents import AGENT_REGISTRY, BaseAgent, make_agent
from .kb import KnowledgeBase
from .world import WumpusWorld

MAX_STEPS_PER_EPISODE = 200


def run_episode(
    agent: BaseAgent,
    size: int = 4,
    pits: int = 3,
    seed: Optional[int] = None,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    trace: bool = False,
) -> Dict:
    """Play one full game. Returns the outcome, and optionally a step-by-step
    trace suitable for animating in the browser."""
    world = WumpusWorld(size=size, pits_count=pits, seed=seed)
    kb = KnowledgeBase(size=world.size, pit_prior=world.pit_prior)
    agent.reset()

    steps: List[Dict] = []
    for _ in range(max_steps):
        percept = world.percept().as_dict()
        kb.tell(world.agent, percept, wumpus_dead=not world.wumpus_alive)

        obs = {
            "pos": world.agent,
            "start": world.start,
            "percept": percept,
            "has_gold": world.has_gold,
            "has_arrow": world.has_arrow,
            "size": world.size,
        }
        decision = agent.decide(kb, obs)
        before = world.agent

        if trace:
            assessment = kb.assess()
            steps.append(
                {
                    "n": len(steps) + 1,
                    "pos": list(before),
                    "percept": percept,
                    "action": decision["action"],
                    "direction": decision.get("direction"),
                    "reason": decision["reason"],
                    "confidence": decision.get("confidence", 1.0),
                    "knowledge": assessment,
                    "score": world.score,
                }
            )

        world.act(decision["action"], decision.get("direction"))
        if world.game_over:
            break

    result = {
        "seed": seed,
        "won": world.won,
        "outcome": world.outcome or "timeout",
        "score": world.score,
        "steps": world.steps,
        "survived": world.outcome not in ("pit", "wumpus"),
        "explored": len(world.visited),
    }
    if trace:
        result["trace"] = steps
        result["final_state"] = world.state(reveal=True)
    return result


def benchmark(
    agent_names: Sequence[str],
    episodes: int = 100,
    size: int = 4,
    pits: int = 3,
    base_seed: int = 20250,
) -> Dict:
    """Paired comparison: every agent faces the identical sequence of caves."""
    seeds = [base_seed + i for i in range(episodes)]
    rows: List[Dict] = []

    for name in agent_names:
        if name not in AGENT_REGISTRY:
            continue
        agent = make_agent(name, seed=base_seed)
        results = [run_episode(agent, size=size, pits=pits, seed=s) for s in seeds]

        scores = [r["score"] for r in results]
        wins = sum(1 for r in results if r["won"])
        deaths = sum(1 for r in results if not r["survived"])
        # A "timeout" means the agent was still wandering after MAX_STEPS - it
        # never decided to leave. Reflex agents livelock this way constantly,
        # oscillating between two squares; planning agents never do.
        stuck = sum(1 for r in results if r["outcome"] == "timeout")
        rows.append(
            {
                "agent": name,
                "label": AGENT_REGISTRY[name].label,
                "episodes": len(results),
                "win_rate": round(100 * wins / len(results), 1),
                "death_rate": round(100 * deaths / len(results), 1),
                "stuck_rate": round(100 * stuck / len(results), 1),
                "survival_rate": round(100 * (len(results) - deaths) / len(results), 1),
                "avg_score": round(statistics.fmean(scores), 1),
                "median_score": round(statistics.median(scores), 1),
                "best_score": max(scores),
                "worst_score": min(scores),
                "avg_steps": round(statistics.fmean([r["steps"] for r in results]), 1),
                "avg_explored": round(statistics.fmean([r["explored"] for r in results]), 1),
            }
        )

    return {
        "config": {"episodes": episodes, "size": size, "pits": pits, "base_seed": base_seed},
        "rows": rows,
    }
