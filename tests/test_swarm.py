import json
import os
import sys
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swarm import MLP, Node, make_dataset, split_non_iid  # noqa: E402


def post(url, obj, sig=""):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Swarm-Sig": sig})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


class SwarmTest(unittest.TestCase):
    def setUp(self):
        self.nodes = []

    def tearDown(self):
        for n in self.nodes:
            n.stop()

    def test_swarm_beats_isolated_peers(self):
        X, y = make_dataset(n=3000, seed=1)
        Xt, yt = make_dataset(n=1000, seed=2)
        shards, _ = split_non_iid(X, y, 4)
        prev = None
        for Xs, ys in shards:
            n = Node(Xs, ys, peers=[prev] if prev else [], round_interval=0.01, secret="k")
            self.nodes.append(n.start())
            prev = n.address
        time.sleep(4)
        accs = [n.model.accuracy(Xt, yt) for n in self.nodes]
        self.assertGreater(min(accs), 0.8, accs)
        self.assertTrue(all(len(n.peers) == 3 for n in self.nodes), "membership did not spread")

    def test_malformed_and_poisoned_updates(self):
        X, y = make_dataset(n=200)
        n = Node(X, y, max_drift=1.0).start(gossip=False)
        self.nodes.append(n)
        before = n.model.copy_params()
        post(n.address + "/gossip", {"weights": {"W1": [[1.0]]}})            # wrong shape
        bad = {k: (np.full_like(v, np.nan)).tolist() for k, v in before.items()}
        post(n.address + "/gossip", {"weights": bad})                          # NaNs
        self.assertEqual(n.rejected, 2)
        huge = {k: (v + 1e6).tolist() for k, v in before.items()}
        post(n.address + "/gossip", {"weights": huge})                         # poison: clipped
        drift = np.sqrt(sum(((n.model.params[k] - before[k]) ** 2).sum() for k in before))
        self.assertLessEqual(drift, 0.5 + 1e-9)

    def test_signature_required_when_secret_set(self):
        X, y = make_dataset(n=200)
        n = Node(X, y, secret="swarm-key").start(gossip=False)
        self.nodes.append(n)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            post(n.address + "/gossip", {"weights": {}}, sig="forged")
        self.assertEqual(cm.exception.code, 401)
        other = Node(X, y, secret="wrong-key").start(gossip=False)
        self.nodes.append(other)
        self.assertFalse(other.gossip_with(n.address))

    def test_predict_endpoint(self):
        X, y = make_dataset(n=500)
        n = Node(X, y).start(gossip=False)
        self.nodes.append(n)
        out = post(n.address + "/predict", {"inputs": [[3, 0], [0, 3]]})
        self.assertEqual(len(out["predictions"]), 2)
        self.assertAlmostEqual(sum(out["probabilities"][0]), 1, places=3)


if __name__ == "__main__":
    unittest.main()
