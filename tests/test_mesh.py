import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh import wire  # noqa: E402
from mesh.aggregate import dp_epsilon, robust_aggregate  # noqa: E402
from mesh.chain import Chain, make_tx  # noqa: E402
from mesh.crypto import Identity  # noqa: E402
from mesh.model import Spec, init_piece, param_hash, slice_backward, slice_forward  # noqa: E402
from mesh.network import Rejected  # noqa: E402
from mesh.sim import World  # noqa: E402


class Pieces(unittest.TestCase):
    def test_slice_backward_matches_numerical_gradient(self):
        spec = Spec()
        p = init_piece(spec, "slice0")
        rng = np.random.default_rng(0)
        h, g = rng.normal(size=(5, spec.hidden)), rng.normal(size=(5, spec.hidden))
        grads, g_in = slice_backward(p, h, g)
        f = lambda: float((slice_forward(p, h) * g).sum())
        eps = 1e-6
        p["W0"][3, 4] += eps
        up = f()
        p["W0"][3, 4] -= 2 * eps
        down = f()
        p["W0"][3, 4] += eps
        self.assertAlmostEqual(grads["W0"][3, 4], (up - down) / (2 * eps), places=5)
        h[1, 2] += eps
        up = f()
        h[1, 2] -= 2 * eps
        self.assertAlmostEqual(g_in[1, 2], (up - f()) / (2 * eps), places=5)

    def test_robust_aggregate_drops_far_outlier_and_caps_size(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=50)
        honest = [{"w": base + rng.normal(0, 0.1, 50)} for _ in range(4)]
        evil = {"w": -100 * base}
        agg, kept = robust_aggregate(honest + [evil], [1] * 5, max_norm=1e9)
        self.assertNotIn(4, kept)
        self.assertGreater(np.dot(agg["w"], base), 0)

    def test_wire_roundtrip_has_no_pickle(self):
        a = {"x": np.arange(6.0).reshape(2, 3)}
        meta, out = wire.unpack(wire.pack({"k": 1}, a))
        self.assertEqual(meta, {"k": 1})
        np.testing.assert_array_equal(out["x"], a["x"])

    def test_privacy_budget_shrinks_with_more_noise(self):
        self.assertGreater(dp_epsilon(1.0, 4, 100), dp_epsilon(5.0, 4, 100))


class Ledger(unittest.TestCase):
    def setUp(self):
        self.vals = [Identity(f"v{i}") for i in range(4)]
        self.alice, self.bob = Identity("alice"), Identity("bob")
        spec = Spec()
        self.genesis = {"spec": spec.to_obj(), "piece_hashes": {p: param_hash(init_piece(spec, p)) for p in spec.pieces},
                        "balances": {self.alice.address: 1000, self.bob.address: 50},
                        "params": {"min_stake": 100, "price": 1, "slash_fraction": 0.5,
                                   "whistleblower_share": 0.2, "emission": 30, "accept_tol": 0.0}}

    def reg(self, who, stake):
        return make_tx(who, "register", {"role": "client", "piece": "ends", "stake": stake,
                                         "sign_pk": who.sign_pk, "kex_pk": who.kex_pk}, 0)

    def test_dishonest_validator_cannot_mint_or_stall(self):
        chain = Chain(self.genesis, self.vals, corrupt={self.vals[0].address})
        chain.submit(self.reg(self.alice, 200))
        chain.produce_block()                     # validator 0 proposes first and is refused
        self.assertEqual(chain.rejected_proposals, 1)
        st = chain.state
        self.assertEqual(st.balances[self.alice.address], 800)
        self.assertNotIn(self.vals[0].address, st.balances)
        self.assertTrue(all(v.state.root() == st.root() for v in chain.validators))

    def test_sybil_needs_stake_and_forged_tx_rejected(self):
        chain = Chain(self.genesis, self.vals)
        chain.submit(self.reg(self.bob, 50))      # below minimum stake
        forged = self.reg(self.alice, 200)
        forged["body"]["stake"] = 999             # tampered after signing
        chain.submit(forged)
        b = chain.produce_block()
        self.assertEqual(b["results"], ["stake too small or unfunded", "bad signature"])
        self.assertEqual(chain.state.nodes, {})


class Privacy(unittest.TestCase):
    def test_vote_is_strict_when_privacy_noise_is_on(self):
        self.assertEqual(World(adversaries=False, dp_sigma=1.0).chain.state.params["accept_tol"], 0.0)
        self.assertEqual(World(adversaries=False).chain.state.params["accept_tol"], 0.25)


class Agreement(unittest.TestCase):
    def test_privacy_noise_does_not_stop_agents_agreeing(self):
        w = World(adversaries=False, activation_noise=0.5)
        for _ in range(6):
            w.round()
        self.assertTrue(any(h["accepted"] for h in w.chain.state.history))


class WholeSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World(adversaries=True)
        cls.taps = []
        cls.w.net.taps.append(cls.taps.append)
        for _ in range(40):
            cls.w.round()

    def test_learns_despite_attackers(self):
        self.assertGreater(self.w.accuracy(), 0.8)

    def test_liar_is_caught_and_starved(self):
        st = self.w.chain.state
        liar = next(s for s in self.w.servers.values() if s.behaviour == "liar")
        self.assertLess(st.stakes[liar.address], 200)
        self.assertLess(st.nodes[liar.address]["rep"], 5)

    def test_poisoned_server_is_outvoted_every_round(self):
        st = self.w.chain.state
        poisoner = next(s for s in self.w.servers.values() if s.behaviour == "poisoner")
        rounds = [h for h in st.history if "slice2" in h["trimmed"]]
        self.assertTrue(rounds)
        self.assertTrue(all(poisoner.address in h["trimmed"]["slice2"] for h in rounds))
        self.assertEqual(st.nodes[poisoner.address]["rep"], 10)   # reputation only grows with training rewards

    def test_honest_replicas_compute_identical_updates(self):
        st = self.w.chain.state
        honest = {s.address for s in self.w.servers.values() if s.behaviour == "honest" and s.online}
        for h in st.history:
            for p in ("slice0", "slice1"):
                self.assertFalse(honest & set(h["trimmed"].get(p, [])), f"honest replica outvoted in {p}")

    def test_poisoning_agent_gets_zero_trust(self):
        st = self.w.chain.state
        cheat = next(a for a in self.w.agents if a.behaviour == "poisoner")
        rounds = [h for h in st.history if "ends" in h["trimmed"]]
        excluded = sum(cheat.address in h["trimmed"]["ends"] for h in rounds)
        self.assertGreaterEqual(excluded, 0.9 * len(rounds))

    def test_traffic_is_unreadable_and_tampering_is_detected(self):
        blob = b"".join(self.taps)
        self.assertEqual(blob.count(b"NUMPY") + blob.count(b"receipt"), 0)
        a, s = self.w.agents[0], self.w.servers[(0, 2)]
        env = a.seal(s.address, "fwd", {"version": s.version}, {"h": np.zeros((1, 32))})
        bad = env[:-5] + bytes([env[-5] ^ 1]) + env[-4:]
        with self.assertRaises(Rejected):
            s.receive(bad)
        s.receive(env)
        with self.assertRaises(Rejected):          # replay
            s.receive(env)

    def test_unregistered_stranger_is_refused(self):
        from mesh.network import Peer
        stranger = Peer(Identity("stranger"))
        self.w.net.attach(stranger)
        s = self.w.servers[(0, 2)]
        env = stranger.seal(s.address, "fwd", {"version": s.version}, {"h": np.zeros((1, 32))})
        with self.assertRaises(Rejected):
            s.receive(env)

    def test_job_market_catches_fake_results(self):
        w = self.w
        poster, cheat, honest = w.agents[0], w.agents[-1], w.agents[1]
        poster.post_job("j", w.Xtest[:40], n_chunks=2, reward=10)
        w.chain.produce_block()
        honest.claim("j", 0)
        cheat.claim("j", 1)
        w.chain.produce_block()
        honest.work_chunk("j", 0)
        cheat.work_chunk("j", 1)
        w.chain.produce_block()
        before = w.chain.state.stakes[cheat.address]
        self.assertEqual(poster.review("j", 0), "paid")
        self.assertEqual(poster.review("j", 1), "disputed")
        w.chain.produce_block()
        st = w.chain.state
        self.assertLess(st.stakes[cheat.address], before)
        self.assertEqual([c["state"] for c in st.jobs["j"]["chunks"]], ["paid", "open"])


if __name__ == "__main__":
    unittest.main()
