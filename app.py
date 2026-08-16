"""
Flask server.

Rules live in wumpus_ai.world, inference in wumpus_ai.kb and wumpus_ai.logic,
policies in wumpus_ai.agents. The routes here just keep one (world, kb, agent)
per browser session and make sure the kb is told every percept once, whether
the human or the agent made the move.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request, session

from wumpus_ai import rl
from wumpus_ai.agents import AGENT_REGISTRY, agent_catalogue, make_agent
from wumpus_ai.benchmark import benchmark
from wumpus_ai.kb import KnowledgeBase
from wumpus_ai.world import WumpusWorld

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "wumpus-dev-key-change-in-production")

# session id -> {"world", "kb", "agent", "agent_name"}
# Ordered so the oldest session can be evicted individually. The previous
# version cleared the whole dict when it filled up, which threw away every
# other player's game to make room for one.
SESSIONS: "OrderedDict[str, Dict]" = OrderedDict()
MAX_SESSIONS = 500
SESSION_TTL = 6 * 60 * 60  # seconds

# The dev server is threaded, so two requests for the same session can mutate
# the same world at once. One lock is plenty at this scale.
STORE_LOCK = threading.Lock()

DIFFICULTIES = {
    "easy": {"size": 4, "pits": 2, "label": "Easy"},
    "classic": {"size": 4, "pits": 3, "label": "Classic 4x4"},
    "hard": {"size": 6, "pits": 6, "label": "6x6"},
    "nightmare": {"size": 8, "pits": 12, "label": "8x8"},
}


# --------------------------------------------------------------------- session
def _session_id() -> str:
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return sid


def _new_game(size: int, pits: int, seed: Optional[int], agent_name: str = "probabilistic") -> Dict:
    world = WumpusWorld(size=size, pits_count=pits, seed=seed)
    kb = KnowledgeBase(size=world.size, pit_prior=world.pit_prior, start=world.start)
    bundle = {
        "world": world,
        "kb": kb,
        "agent": make_agent(agent_name),
        "agent_name": agent_name,
        "touched": time.time(),
    }
    _observe(bundle)
    return bundle


def _evict() -> None:
    """Drop expired sessions, then the oldest if we are still over the cap."""
    cutoff = time.time() - SESSION_TTL
    for sid in [s for s, b in SESSIONS.items() if b.get("touched", 0) < cutoff]:
        SESSIONS.pop(sid, None)
    while len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.popitem(last=False)


def _store(sid: str, bundle: Dict) -> Dict:
    _evict()
    SESSIONS[sid] = bundle
    SESSIONS.move_to_end(sid)
    return bundle


def _get_bundle() -> Dict:
    sid = _session_id()
    bundle = SESSIONS.get(sid)
    if bundle is None:
        return _store(sid, _new_game(4, 3, None))
    bundle["touched"] = time.time()
    SESSIONS.move_to_end(sid)
    return bundle


def _observe(bundle: Dict) -> None:
    """TELL the knowledge base what the agent's senses report right now."""
    world: WumpusWorld = bundle["world"]
    kb: KnowledgeBase = bundle["kb"]
    kb.tell(world.agent, world.percept().as_dict(), wumpus_dead=not world.wumpus_alive)


def _payload(bundle: Dict, decision: Optional[Dict] = None, reveal: bool = False) -> Dict:
    world: WumpusWorld = bundle["world"]
    kb: KnowledgeBase = bundle["kb"]
    data = {
        "state": world.state(reveal=reveal),
        "knowledge": kb.assess(),
        "agent_name": bundle["agent_name"],
    }
    if decision is not None:
        data["decision"] = decision
    return data


def _observation(bundle: Dict) -> Dict:
    world: WumpusWorld = bundle["world"]
    return {
        "pos": world.agent,
        "start": world.start,
        "percept": world.percept().as_dict(),
        "has_gold": world.has_gold,
        "has_arrow": world.has_arrow,
        "size": world.size,
    }


def _ensure_agent(bundle: Dict, name: Optional[str]) -> None:
    if name and name in AGENT_REGISTRY and name != bundle["agent_name"]:
        bundle["agent"] = make_agent(name)
        bundle["agent_name"] = name


def _body() -> Dict:
    return request.get_json(silent=True) or {}


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def _parse_seed(value) -> Optional[int]:
    """Anything unparseable means "surprise me" rather than a 500.

    Bounded rather than masked: masking turned a typed -5 into 2147483643,
    which is then echoed back in the seed box and looks like a bug.
    """
    if value in (None, "", "random"):
        return None
    try:
        return max(-(2 ** 31), min(2 ** 31 - 1, int(value)))
    except (TypeError, ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    return jsonify(
        {
            "agents": agent_catalogue(),
            "difficulties": DIFFICULTIES,
            "scoring": {
                "gold_escape": 1000,
                "death": -1000,
                "arrow": -10,
                "action": -1,
            },
        }
    )


@app.route("/api/new", methods=["POST"])
def api_new():
    data = _body()
    size = _clamp(data.get("size", 4), 3, 12, 4)
    pits = _clamp(data.get("pits", 3), 0, size * size - 2, 3)
    seed = _parse_seed(data.get("seed"))
    agent_name = data.get("agent") or "probabilistic"

    with STORE_LOCK:
        bundle = _store(_session_id(), _new_game(size, pits, seed, agent_name))
        return jsonify(_payload(bundle))


@app.route("/api/state")
def api_state():
    with STORE_LOCK:
        return jsonify(_payload(_get_bundle()))


@app.route("/api/action", methods=["POST"])
def api_action():
    data = _body()
    with STORE_LOCK:
        bundle = _get_bundle()
        world: WumpusWorld = bundle["world"]
        world.act(data.get("action", ""), data.get("direction"))
        _observe(bundle)
        return jsonify(_payload(bundle))


@app.route("/api/hint", methods=["POST"])
def api_hint():
    """What the agent would do here. Advice only, nothing is executed."""
    data = _body()
    with STORE_LOCK:
        bundle = _get_bundle()
        _ensure_agent(bundle, data.get("agent"))
        if bundle["world"].game_over:
            return jsonify(_payload(bundle, decision={"action": "none", "reason": "The game is over."}))
        decision = bundle["agent"].decide(bundle["kb"], _observation(bundle))
        return jsonify(_payload(bundle, decision=decision))


@app.route("/api/ai/step", methods=["POST"])
def api_ai_step():
    data = _body()
    with STORE_LOCK:
        bundle = _get_bundle()
        _ensure_agent(bundle, data.get("agent"))
        world: WumpusWorld = bundle["world"]
        if world.game_over:
            return jsonify(_payload(bundle))

        decision = bundle["agent"].decide(bundle["kb"], _observation(bundle))
        world.act(decision["action"], decision.get("direction"))
        _observe(bundle)
        return jsonify(_payload(bundle, decision=decision))


@app.route("/api/ai/solve", methods=["POST"])
def api_ai_solve():
    """Play the current game out to the end, returning every step for replay."""
    data = _body()
    max_steps = _clamp(data.get("max_steps", 120), 1, 400, 120)

    with STORE_LOCK:
        bundle = _get_bundle()
        _ensure_agent(bundle, data.get("agent"))
        world: WumpusWorld = bundle["world"]

        frames = []
        for _ in range(max_steps):
            if world.game_over:
                break
            decision = bundle["agent"].decide(bundle["kb"], _observation(bundle))
            world.act(decision["action"], decision.get("direction"))
            _observe(bundle)
            frames.append(_payload(bundle, decision=decision))

        return jsonify({"frames": frames, "final": _payload(bundle)})


@app.route("/api/reveal", methods=["POST"])
def api_reveal():
    with STORE_LOCK:
        return jsonify(_payload(_get_bundle(), reveal=True))


@app.route("/api/proof", methods=["POST"])
def api_proof():
    """Resolution proof for what the CNF knowledge base can show about a square."""
    data = _body()
    with STORE_LOCK:
        bundle = _get_bundle()
        size = bundle["world"].size
        r = _clamp(data.get("r", 0), 0, size - 1, 0)
        c = _clamp(data.get("c", 0), 0, size - 1, 0)
        return jsonify(bundle["kb"].prove((r, c)))


@app.route("/api/rl/train", methods=["POST"])
def api_rl_train():
    """Train a fresh Q-table in the browser and hand back the learning curve.

    Capped well below what the bundled table was trained on: the point here is
    to watch the curve climb, not to beat the shipped agent inside one request.
    """
    data = _body()
    episodes = _clamp(data.get("episodes", 12000), 500, 40000, 12000)
    size = _clamp(data.get("size", 4), 3, 6, 4)
    pits = _clamp(data.get("pits", 3), 0, size * size - 2, 3)

    # Measured at ~2.1ms/episode on 4x4 and ~4.8ms/episode on 6x6 (roughly
    # linear in board area). Uncapped, 40000 episodes on 6x6 took 192s on one
    # synchronous request - long past any sane HTTP timeout. Same wall-clock
    # budget approach as /api/benchmark: trim and say so, rather than hang.
    requested = episodes
    UNIT_BUDGET = 7000  # episodes, measured at 4x4
    scale = max(1.0, (size * size) / 16.0)
    allowed = int(UNIT_BUDGET / scale)
    episodes = max(500, min(episodes, allowed))

    result = rl.train(episodes=episodes, size=size, pits=pits)
    shipped = rl.load()
    return jsonify({
        "capped_from": requested if episodes < requested else None,
        "curve": result["curve"],
        "episodes": result["episodes"],
        "states_seen": result["states_seen"],
        "config": result["config"],
        "shipped_episodes": shipped.trained_episodes if shipped else 0,
    })


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = _body()
    agents = [a for a in data.get("agents", []) if a in AGENT_REGISTRY]
    if not agents:
        agents = list(AGENT_REGISTRY)
    episodes = _clamp(data.get("episodes", 100), 5, 1000, 100)
    size = _clamp(data.get("size", 4), 3, 10, 4)
    pits = _clamp(data.get("pits", 3), 0, size * size - 2, 3)

    # The whole run happens inside one request, so it needs a wall-clock budget
    # rather than a raw episode cap. Measured at roughly 8ms per episode-agent
    # on 4x4, so ~1500 of them is about twelve seconds; bigger boards cost
    # proportionally more. Trim to fit and tell the caller, rather than
    # silently running a different experiment than the one they asked for.
    requested = episodes
    UNIT_BUDGET = 1500  # episode-agent runs, measured at 4x4
    scale = max(1.0, (size * size) / 16.0)
    allowed = int(UNIT_BUDGET / (len(agents) * scale))
    episodes = max(5, min(episodes, allowed))

    result = benchmark(agents, episodes=episodes, size=size, pits=pits)
    if episodes < requested:
        result["capped_from"] = requested
    return jsonify(result)


@app.errorhandler(500)
def on_error(err):  # pragma: no cover - safety net for the demo
    app.logger.exception("unhandled error")
    return jsonify({"error": "internal error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Off by default. The Werkzeug debugger is an interactive shell, so leaving
    # it on in a repo people clone and run is not something to do by accident.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    print(f"\n  Wumpus World  ->  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=debug)
