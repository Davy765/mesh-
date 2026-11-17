import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh.sim import World  # noqa: E402


def withholding_world(relay, rounds=4):
    """Two colluding agents each send their slice-0 training input to a DIFFERENT single replica and refuse to
    hand it out, trying to split the three replicas three ways so no update ever gets a majority."""
    w = World(adversaries=False)
    s0 = sorted((w.servers[(0, r)] for r in range(3)), key=lambda s: s.address)
    a, b = w.agents[0], w.agents[1]
    a.behaviour = b.behaviour = "withholder"
    a.withhold_to, b.withhold_to = {s0[0].address}, {s0[1].address}
    for s in w.servers.values():
        s.peer_relay = relay
    for _ in range(rounds):
        w.round()
    return w, s0


class Withholding(unittest.TestCase):
    def test_without_relay_the_attack_stalls_training(self):
        w, _ = withholding_world(relay=False)
        hist = w.chain.state.history
        self.assertTrue(all(not h["accepted"] for h in hist))
        self.assertTrue(all("slice0" not in h["trimmed"] for h in hist), "slice 0 never reached a majority")

    def test_with_relay_replicas_agree_and_training_continues(self):
        w, s0 = withholding_world(relay=True)
        hist = w.chain.state.history
        self.assertTrue(any(h["accepted"] for h in hist))
        honest = {s.address for s in s0}
        for h in hist:
            self.assertIn("slice0", h["trimmed"])                       # a majority was reached...
            self.assertFalse(honest & set(h["trimmed"]["slice0"]))      # ...and no honest replica was outvoted

    def test_salt_withholding_server_cannot_get_an_honest_replica_outvoted(self):
        for relay in (False, True):
            w = World(adversaries=False, dp_sigma=0.3)
            bad = w.servers[(0, 0)]
            bad.behaviour = "withholder"
            for s in w.servers.values():
                s.peer_relay = relay
            for _ in range(3):
                w.round()
            honest = {w.servers[(0, r)].address for r in (1, 2)}
            outvoted = set().union(*(h["trimmed"].get("slice0", []) for h in w.chain.state.history))
            if relay:
                self.assertFalse(honest & outvoted)
            else:
                self.assertTrue(honest & outvoted, "without relay the left-out replica should lose")


if __name__ == "__main__":
    unittest.main()
