"""
Training loop for the Q-learning agent, plus saving and loading the table.

The reward signal is just the environment's own score delta, so the agent is
optimising exactly the number the other agents are scored on. Each episode gets
a fresh random cave, which forces it to learn a policy rather than memorise one
map.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .agents import QLearningAgent
from .kb import KnowledgeBase
from .world import WumpusWorld

TABLE_PATH = Path(__file__).with_name("q_table.json")
MAX_STEPS = 120


def train(
    episodes: int = 20000,
    size: int = 4,
    pits: int = 3,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    agent: Optional[QLearningAgent] = None,
    checkpoints: int = 40,
    seed: int = 0,
    timeout_penalty: float = -300.0,
) -> Dict:
    """Run `episodes` of Q-learning and return the agent plus a learning curve."""
    agent = agent or QLearningAgent(alpha=alpha, gamma=gamma, seed=seed)
    agent.alpha = alpha
    agent.gamma = gamma
    rng = random.Random(seed)

    block = max(1, episodes // max(1, checkpoints))
    curve: List[Dict] = []
    window: List[int] = []

    for episode in range(episodes):
        # Decay to the floor over the first 60% of training, then hold there,
        # so the last 40% is spent refining the policy it actually intends to
        # use rather than still wandering off at random.
        progress = min(1.0, episode / max(1, episodes * 0.6))
        agent.epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        world = WumpusWorld(size=size, pits_count=pits, seed=rng.randrange(1 << 30))
        kb = KnowledgeBase(world.size, world.pit_prior, start=world.start)
        kb.cnf = None  # the prover is dead weight during training
        agent.reset()

        previous_score = 0
        for _ in range(MAX_STEPS):
            percept = world.percept().as_dict()
            kb.tell(world.agent, percept, wumpus_dead=not world.wumpus_alive)
            obs = {
                "pos": world.agent, "start": world.start, "percept": percept,
                "has_gold": world.has_gold, "has_arrow": world.has_arrow, "size": world.size,
            }
            key = agent.state_key(kb, obs)
            index = agent.choose(key)
            action, direction = agent.ACTIONS[index]

            world.act(action, direction)
            reward = world.score - previous_score
            previous_score = world.score

            if world.game_over:
                agent.update(key, index, reward, None)
                break

            next_percept = world.percept().as_dict()
            kb.tell(world.agent, next_percept, wumpus_dead=not world.wumpus_alive)
            next_obs = dict(obs)
            next_obs.update({"pos": world.agent, "percept": next_percept, "has_gold": world.has_gold})
            agent.update(key, index, reward, agent.state_key(kb, next_obs))
        else:
            # Ran out of steps without ever deciding to leave. Without a real
            # penalty here the agent settles into shuffling between two safe
            # squares forever, which never dies and so never looks bad.
            agent.update(key, index, timeout_penalty, None)

        window.append(world.score)
        agent.trained_episodes += 1

        if (episode + 1) % block == 0:
            curve.append({
                "episode": episode + 1,
                "avg_score": round(sum(window) / len(window), 1),
                "epsilon": round(agent.epsilon, 3),
            })
            window = []

    agent.epsilon = 0.0
    return {
        "agent": agent,
        "curve": curve,
        "episodes": episodes,
        "states_seen": len(agent.q),
        "config": {"size": size, "pits": pits, "alpha": alpha, "gamma": gamma},
    }


def save(agent: QLearningAgent, path: Path = TABLE_PATH) -> None:
    payload = {
        "trained_episodes": agent.trained_episodes,
        # JSON keys must be strings, so the state tuple is flattened with '|'.
        "table": {_encode(k): v for k, v in agent.q.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load(path: Path = TABLE_PATH) -> Optional[QLearningAgent]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    agent = QLearningAgent(table={_decode(k): v for k, v in payload["table"].items()})
    agent.trained_episodes = payload.get("trained_episodes", 0)
    return agent


def _encode(key) -> str:
    breeze, stench, glitter, gold, at_start, around = key
    flags = "".join("1" if f else "0" for f in (breeze, stench, glitter, gold, at_start))
    return flags + "|" + "".join(str(a) for a in around)


def _decode(text: str):
    flags, around = text.split("|")
    return (
        flags[0] == "1", flags[1] == "1", flags[2] == "1", flags[3] == "1", flags[4] == "1",
        tuple(int(ch) for ch in around),
    )


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    result = train(episodes=count)
    save(result["agent"])
    print(f"trained {count} episodes, {result['states_seen']} states, saved to {TABLE_PATH.name}")
    for point in result["curve"][::4]:
        print(f"  ep {point['episode']:>6}  avg score {point['avg_score']:>8}  eps {point['epsilon']}")
