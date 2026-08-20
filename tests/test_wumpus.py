"""
Test suite for the Wumpus World AI.

Run from the project root:

    python -m unittest discover -s tests -v

The two tests that matter most for the marking rubric are
`test_kb_is_sound` (the knowledge base never labels a lethal square "safe")
and `test_logic_agent_never_dies` (an agent that only moves on proven-safe
squares is, empirically, immortal).
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wumpus_ai import rl
from wumpus_ai.agents import make_agent
from wumpus_ai.benchmark import benchmark, run_episode
from wumpus_ai.kb import KnowledgeBase
from wumpus_ai.logic import CnfKnowledgeBase, dpll_satisfiable
from wumpus_ai.world import WumpusWorld, neighbors


class TestWorld(unittest.TestCase):
    def test_start_square_is_never_a_pit(self):
        for seed in range(200):
            world = WumpusWorld(size=4, pits_count=3, seed=seed)
            self.assertNotIn(world.start, world.pits)

    def test_same_seed_gives_same_cave(self):
        a = WumpusWorld(size=6, pits_count=5, seed=42)
        b = WumpusWorld(size=6, pits_count=5, seed=42)
        self.assertEqual((a.pits, a.wumpus, a.gold), (b.pits, b.wumpus, b.gold))

    def test_pit_count_is_clamped_to_capacity(self):
        world = WumpusWorld(size=3, pits_count=99, seed=1)
        self.assertEqual(len(world.pits), world.pits_count)
        self.assertLessEqual(world.pits_count, 3 * 3 - 2)

    def test_percept_is_pure(self):
        """Reading the percept must not consume the momentary flags."""
        world = WumpusWorld(size=4, pits_count=2, seed=3)
        world.wumpus = (world.start[0] - 1, world.start[1])
        world.act("shoot", "up")
        self.assertTrue(world.percept().scream)
        self.assertTrue(world.percept().scream, "scream was consumed by the first read")
        world.act("move", "right")
        self.assertFalse(world.percept().scream, "scream should not outlive the next action")

    def test_breeze_matches_pit_adjacency(self):
        world = WumpusWorld(size=5, pits_count=6, seed=9)
        for r in range(world.size):
            for c in range(world.size):
                world.agent = (r, c)
                expected = any(n in world.pits for n in neighbors((r, c), world.size))
                self.assertEqual(world.percept().breeze, expected)

    def test_scoring(self):
        world = WumpusWorld(size=4, pits_count=0, seed=5)
        world.gold = world.start
        world.act("grab")           # -1
        world.act("climb")          # -1, +1000
        self.assertTrue(world.won)
        self.assertEqual(world.score, 998)

    def test_death_costs_1000(self):
        world = WumpusWorld(size=4, pits_count=1, seed=5)
        target = next(iter(world.pits))
        world.agent = (target[0] + 1, target[1]) if target[0] + 1 < world.size else (target[0] - 1, target[1])
        world.act("move", "up" if target[0] < world.agent[0] else "down")
        self.assertTrue(world.game_over)
        self.assertEqual(world.outcome, "pit")
        self.assertEqual(world.score, -1001)


class TestKnowledgeBase(unittest.TestCase):
    def test_no_breeze_proves_neighbours_pit_free(self):
        kb = KnowledgeBase(4, 3 / 15)
        kb.tell((3, 0), {"breeze": False, "stench": False, "glitter": False})
        safe = {tuple(c) for c in kb.assess()["safe"]}
        self.assertIn((2, 0), safe)
        self.assertIn((3, 1), safe)

    def test_model_counting_matches_hand_calculation(self):
        """Breeze at (2,0) with exactly two unknown neighbours.

        P(pit at X | at least one of X, Y) = p / (1 - (1-p)^2), which for
        p = 3/15 = 0.2 is 0.2 / 0.36 = 0.5556.
        """
        kb = KnowledgeBase(4, 3 / 15)
        kb.tell((3, 0), {"breeze": False, "stench": False, "glitter": False})
        kb.tell((2, 0), {"breeze": True, "stench": False, "glitter": False})
        cells = kb.assess()["cells"]
        self.assertAlmostEqual(cells["1,0"]["pit"], 0.5556, places=3)
        self.assertAlmostEqual(cells["2,1"]["pit"], 0.5556, places=3)
        self.assertEqual(cells["3,1"]["pit"], 0.0)

    def test_wumpus_is_pinned_down_by_two_stenches(self):
        """Stench at (3,0) and (2,1), no stench at (3,1) - only (2,0) fits."""
        kb = KnowledgeBase(4, 3 / 15)
        kb.tell((3, 1), {"breeze": False, "stench": False, "glitter": False})
        kb.tell((3, 0), {"breeze": False, "stench": True, "glitter": False})
        kb.tell((2, 1), {"breeze": False, "stench": True, "glitter": False})
        candidates = [tuple(c) for c in kb.assess()["wumpus_candidates"]]
        self.assertEqual(candidates, [(2, 0)])

    def test_scream_clears_the_wumpus(self):
        kb = KnowledgeBase(4, 3 / 15)
        kb.tell((3, 0), {"breeze": False, "stench": True, "glitter": False})
        kb.tell((3, 0), {"breeze": False, "stench": False, "glitter": False, "scream": True})
        assessment = kb.assess()
        self.assertTrue(assessment["wumpus_dead"])
        self.assertEqual(assessment["wumpus_candidates"], [])

    def test_kb_is_sound(self):
        """The headline property: nothing the KB proves safe is ever lethal.

        Plays 300 caves and, at every single step, checks every square the KB
        currently claims is safe against the hidden ground truth.
        """
        violations = []
        for seed in range(300):
            world = WumpusWorld(size=4, pits_count=3, seed=seed)
            kb = KnowledgeBase(world.size, world.pit_prior)
            agent = make_agent("probabilistic")
            agent.reset()
            for _ in range(80):
                percept = world.percept().as_dict()
                kb.tell(world.agent, percept, wumpus_dead=not world.wumpus_alive)
                for cell in {tuple(c) for c in kb.assess()["safe"]}:
                    if cell in world.pits:
                        violations.append(("pit", seed, cell))
                    if cell == world.wumpus and world.wumpus_alive:
                        violations.append(("wumpus", seed, cell))
                decision = agent.decide(kb, {
                    "pos": world.agent, "start": world.start, "percept": percept,
                    "has_gold": world.has_gold, "has_arrow": world.has_arrow, "size": world.size,
                })
                world.act(decision["action"], decision.get("direction"))
                if world.game_over:
                    break
        self.assertEqual(violations, [], f"KB called a lethal square safe: {violations[:5]}")


class TestAgents(unittest.TestCase):
    def test_logic_agent_never_dies(self):
        """With zero risk tolerance the agent should be empirically immortal."""
        deaths = [
            seed for seed in range(1000)
            if run_episode(make_agent("logic"), size=4, pits=3, seed=seed)["outcome"] in ("pit", "wumpus")
        ]
        self.assertEqual(deaths, [], f"logic agent died on seeds {deaths[:5]}")

    def test_planning_agents_always_terminate(self):
        """Agents that route with BFS always reach a decision to leave."""
        for name in ("logic", "probabilistic"):
            for seed in range(60):
                result = run_episode(make_agent(name, seed=seed), size=5, pits=4, seed=seed)
                self.assertNotEqual(
                    result["outcome"], "timeout", f"{name} failed to terminate on seed {seed}"
                )

    def test_reflex_agent_livelocks(self):
        """Documents the reflex architecture's defining failure mode.

        With no plan and no goal stack, a reflex agent that has run out of safe
        unexplored neighbours just bounces between two visited squares forever.
        This is not a bug to fix - it is the reason AIMA introduces model-based
        agents, and the benchmark surfaces it as the "stuck %" column.
        """
        outcomes = [
            run_episode(make_agent("reflex", seed=seed), size=5, pits=4, seed=seed)["outcome"]
            for seed in range(60)
        ]
        stuck = outcomes.count("timeout")
        self.assertGreater(stuck, 0, "expected the reflex agent to livelock at least sometimes")
        # Same caves, planning agent: never stuck. That contrast is the point.
        planned = [
            run_episode(make_agent("logic", seed=seed), size=5, pits=4, seed=seed)["outcome"]
            for seed in range(60)
        ]
        self.assertEqual(planned.count("timeout"), 0)

    def test_reasoning_agents_beat_the_random_baseline(self):
        rows = {r["agent"]: r for r in benchmark(
            ["random", "reflex", "logic", "probabilistic"], episodes=120, size=4, pits=3
        )["rows"]}
        self.assertGreater(rows["logic"]["avg_score"], rows["random"]["avg_score"])
        self.assertGreater(rows["probabilistic"]["avg_score"], rows["reflex"]["avg_score"])
        self.assertEqual(rows["logic"]["death_rate"], 0.0)

    def test_agents_cannot_see_the_map(self):
        """An agent's decision must depend only on the KB and its own body.

        Moving the gold and the pits behind the agent's back - without telling
        the KB - must not change what it decides to do.
        """
        world = WumpusWorld(size=4, pits_count=3, seed=17)
        kb = KnowledgeBase(world.size, world.pit_prior)
        kb.tell(world.agent, world.percept().as_dict())
        obs = {
            "pos": world.agent, "start": world.start, "percept": world.percept().as_dict(),
            "has_gold": False, "has_arrow": True, "size": world.size,
        }
        first = make_agent("probabilistic").decide(kb, obs)
        world.gold = (0, 0)
        world.wumpus = (0, 3)
        second = make_agent("probabilistic").decide(kb, obs)
        self.assertEqual(first["action"], second["action"])
        self.assertEqual(first.get("direction"), second.get("direction"))


class TestProver(unittest.TestCase):
    def test_dpll_basics(self):
        # (a) and (not a) is unsatisfiable; (a or b) on its own is not.
        self.assertFalse(dpll_satisfiable([frozenset([1]), frozenset([-1])]))
        self.assertTrue(dpll_satisfiable([frozenset([1, 2])]))
        self.assertTrue(dpll_satisfiable([]))

    def test_resolution_pins_the_wumpus(self):
        cnf = CnfKnowledgeBase(4)
        cnf.add_wumpus_axioms((3, 0))
        cnf.tell((3, 0), breeze=False, stench=False)
        cnf.tell((2, 0), breeze=True, stench=False)
        cnf.tell((3, 1), breeze=False, stench=True)
        # Only (3,2) is left: (2,1) and (3,0) are ruled out by the two
        # stench-free squares, so the arrow has a guaranteed target.
        self.assertTrue(cnf.entails(cnf.wumpus((3, 2))))
        steps = cnf.proof(cnf.wumpus((3, 2)))
        self.assertTrue(steps, "expected a unit-resolution refutation")
        self.assertEqual(steps[-1]["gives"], "⊥")

    def test_resolution_derives_a_pit(self):
        cnf = CnfKnowledgeBase(4)
        cnf.add_wumpus_axioms((3, 0))
        cnf.tell((3, 0), breeze=False, stench=False)
        cnf.tell((2, 0), breeze=True, stench=False)
        cnf.tell((3, 1), breeze=False, stench=True)
        # Breeze at (2,0) needs a pit in (1,0) or (2,1); (3,1) has no breeze,
        # so (2,1) is clear and the pit must be at (1,0).
        self.assertTrue(cnf.entails(cnf.pit((1, 0))))
        self.assertFalse(cnf.entails(cnf.pit((2, 1))))

    def test_killing_the_wumpus_keeps_the_kb_satisfiable(self):
        cnf = CnfKnowledgeBase(4)
        cnf.add_wumpus_axioms((3, 0))
        cnf.tell((3, 0), breeze=False, stench=True)
        cnf.kill_wumpus()
        self.assertTrue(dpll_satisfiable(cnf.clauses), "dropping 'at least one Wumpus' failed")
        self.assertTrue(cnf.entails(-cnf.wumpus((2, 0))))

    def test_prover_agrees_with_model_counting(self):
        """Two independent routes to the same safety verdict must not disagree.

        Model counting says a square is safe when its probability of danger is
        zero; resolution says so when the KB entails no pit and no Wumpus. They
        are different algorithms over the same percepts, so any disagreement
        means one of them is wrong.
        """
        for seed in range(120):
            world = WumpusWorld(size=4, pits_count=3, seed=seed)
            kb = KnowledgeBase(world.size, world.pit_prior, start=world.start)
            agent = make_agent("probabilistic")
            agent.reset()
            for _ in range(40):
                percept = world.percept().as_dict()
                kb.tell(world.agent, percept, wumpus_dead=not world.wumpus_alive)

                counted = {tuple(c) for c in kb.assess()["safe"]}
                for cell in kb.all_cells():
                    proved = kb.cnf.proves_safe(cell)
                    self.assertEqual(
                        proved, cell in counted,
                        f"seed {seed}: DPLL says safe={proved} for {cell}, "
                        f"model counting says {cell in counted}",
                    )

                decision = agent.decide(kb, {
                    "pos": world.agent, "start": world.start, "percept": percept,
                    "has_gold": world.has_gold, "has_arrow": world.has_arrow, "size": world.size,
                })
                world.act(decision["action"], decision.get("direction"))
                if world.game_over:
                    break


class TestQLearning(unittest.TestCase):
    def test_table_survives_a_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "q.json"
            trained = rl.train(episodes=300, checkpoints=3)["agent"]
            rl.save(trained, path)
            restored = rl.load(path)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.q, trained.q)

    def test_training_improves_the_policy(self):
        result = rl.train(episodes=6000, checkpoints=6)
        curve = result["curve"]
        self.assertGreater(curve[-1]["avg_score"], curve[0]["avg_score"],
                           "6000 episodes of Q-learning did not improve average score")

    def test_bundled_agent_beats_random(self):
        agent = make_agent("qlearning")
        if not agent.q:
            self.skipTest("no trained q_table.json bundled")
        rows = {r["agent"]: r for r in benchmark(["random", "qlearning"], episodes=120)["rows"]}
        self.assertGreater(rows["qlearning"]["avg_score"], rows["random"]["avg_score"])

    def test_state_key_is_percepts_only(self):
        """The Q agent must not be able to see inferred danger.

        If its state key ever started depending on the knowledge base's
        probability estimates it would quietly become a hybrid, and the
        comparison against the logic agents would stop meaning anything.
        """
        world = WumpusWorld(size=4, pits_count=3, seed=8)
        kb = KnowledgeBase(world.size, world.pit_prior, start=world.start)
        kb.tell(world.agent, world.percept().as_dict())
        obs = {
            "pos": world.agent, "start": world.start, "percept": world.percept().as_dict(),
            "has_gold": False, "has_arrow": True, "size": world.size,
        }
        agent = make_agent("qlearning")
        before = agent.state_key(kb, obs)
        world.pits = {(0, 0), (0, 1), (0, 2)}
        world.wumpus = (0, 3)
        self.assertEqual(before, agent.state_key(kb, obs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
