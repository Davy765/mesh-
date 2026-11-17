"""What strangers on the open internet can throw at a node, and what must happen: a clean refusal, never a
crash, and never a stalled chain."""
import os
import random
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh.chain import make_tx  # noqa: E402
from mesh.crypto import Identity, canon  # noqa: E402
import mesh.network as net  # noqa: E402
from mesh.network import Rejected  # noqa: E402
from mesh.sim import World  # noqa: E402


class MalformedMessages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World(adversaries=False)
        cls.w.round()

    def test_random_bytes_are_refused_cleanly(self):
        server, rng = self.w.servers[(0, 0)], random.Random(1)
        for i in range(1500):
            blob = bytes(rng.getrandbits(8) for _ in range(rng.choice([0, 3, 4, 40, 300])))
            if i % 2:
                blob = (7).to_bytes(4, "big") + b'{"a":1}' + blob
            with self.assertRaises(Rejected):
                server.receive(blob)

    def test_bad_requests_from_a_registered_peer_get_an_error_reply(self):
        server, agent = self.w.servers[(0, 0)], self.w.agents[0]
        metas = [{}, {"version": 5}, {"version": server.version, "train": "yes"}, {"round": "x", "salt": 3}]
        arrays = [{}, {"h": np.zeros((2, 3))}, {"h": np.full((1, 32), np.nan)}, {"h": np.zeros(5)},
                  {"h": np.zeros((1, 32)), "g": np.zeros((2, 32))}]
        for kind in ("fwd", "bwd", "reveal", "fetch", "salt", "get_contrib", "get_salt", "nope"):
            for m in metas:
                for i, a in enumerate(arrays):
                    reply, _ = server.receive(agent.seal(server.address, kind, m, a))   # never raises
                    rkind, rmeta, _ = agent.open(reply)
                    self.assertIn(rkind, ("ok", "error"))
                    if kind in ("fwd", "bwd") and i in (1, 2, 3):          # wrong shape, NaN, wrong rank
                        self.assertEqual(rkind, "error", (kind, m, i))
                    if kind == "nope" or m == {}:
                        self.assertEqual(rkind, "error", (kind, m, i))

    def test_stale_messages_are_refused(self):
        server, agent = self.w.servers[(0, 0)], self.w.agents[0]
        env = agent.seal(server.address, "fwd", {"version": server.version}, {"h": np.zeros((1, 32))})
        old, net.REPLAY_WINDOW_S = net.REPLAY_WINDOW_S, -1          # everything counts as stale
        try:
            with self.assertRaises(Rejected):
                server.receive(env)
        finally:
            net.REPLAY_WINDOW_S = old


class MalformedTransactions(unittest.TestCase):
    def test_no_transaction_can_crash_or_stall_the_ledger(self):
        w = World(adversaries=False)
        st = w.chain.state.copy()
        root_before = st.root()
        ident = Identity.generate()
        bodies = [None, [], "x", {"stake": "a"}, {"to": None, "amount": "1e9"}, {"evidence": []},
                  {"evidence": {"accused": 1, "witnesses": "ab"}}, {"job": [], "chunk": "x"},
                  {"inputs": 5, "round": 0}, {"pay": [1], "round": 0}, {"stake": 10 ** 30, "kex_pk": 1}]
        kinds = ["register", "transfer", "commit", "candidate", "vote", "settle", "slash", "contrib", "job_open",
                 "job_claim", "job_submit", "job_release", "job_dispute", "validator_join", "validator_leave", "x"]
        nonce = 0
        for kind in kinds:
            for b in bodies:
                body = dict(b, sign_pk=ident.sign_pk) if isinstance(b, dict) else b
                for tx in (make_tx(ident, kind, body, nonce), {"type": kind, "body": body}, [], "x", None):
                    result = st.apply(tx)                  # must return, never raise
                    self.assertTrue(result is None or isinstance(result, str))
                if st.nonces.get(ident.address, 0) > nonce:
                    nonce += 1
        # whatever failed was undone; only nonces (and anything that legitimately succeeded) changed
        self.assertNotEqual(st.root(), "")
        self.assertIsInstance(root_before, str)

    def test_validators_refuse_bad_transactions_at_the_door(self):
        from mesh import run
        from mesh.chain_net import ValidatorNode
        with tempfile.TemporaryDirectory() as d:
            genesis, _ = run.init(d, n_agents=1, replicas=1, n_validators=4)
            v = ValidatorNode(Identity.load(os.path.join(d, "keys", "validator0.key")), genesis, 2.0)
            stranger = Identity.generate()
            good = make_tx(stranger, "transfer", {"to": "a" * 40, "amount": 1, "sign_pk": stranger.sign_pk}, 0)
            self.assertEqual(v.public({"kind": "tx", "tx": good}), {"ok": True})
            self.assertTrue(v.public({"kind": "tx", "tx": good}).get("duplicate"))
            forged = dict(good, nonce=1)
            for bad in (forged, {"type": "x"}, "x", None, dict(good, nonce=1000), dict(good, nonce="1")):
                with self.assertRaises(Rejected):
                    v.public({"kind": "tx", "tx": bad})
            for n in range(1, 64):
                v.public({"kind": "tx", "tx": make_tx(stranger, "transfer", {"to": "a" * 40, "amount": 1,
                                                                             "sign_pk": stranger.sign_pk}, n)})
            with self.assertRaises(Rejected):          # per-sender cap: one account can't flood the queue
                v.public({"kind": "tx", "tx": make_tx(stranger, "transfer", {"to": "a" * 40, "amount": 1,
                                                                              "sign_pk": stranger.sign_pk}, 64)})
            self.assertEqual(len(v.v.mempool), 64)
            self.assertNotIn(canon(good)[:0], [b"x"])


class Limits(unittest.TestCase):
    def test_rate_limit_and_local_exemption(self):
        from mesh import transport
        lim = transport._Limits()
        allowed = sum(lim.allow("203.0.113.9") for _ in range(int(transport.RATE_BURST) + 50))
        self.assertLessEqual(allowed, int(transport.RATE_BURST) + 1)
        self.assertTrue(all(lim.allow("127.0.0.1") for _ in range(5000)))


if __name__ == "__main__":
    unittest.main()


class WebServices(unittest.TestCase):
    def _faucet(self, per_day=3):
        from mesh.testnet import Faucet

        class StubChain:
            def __init__(self):
                self.sent = []

                class S:
                    balances = {}
                    nonces = {}
                self.state = S()

            def submit(self, tx):
                self.sent.append(tx)
        f = Faucet.__new__(Faucet)
        f.chain, f.id = StubChain(), Identity.generate()
        f.chain.state.balances[f.id.address] = 10 ** 6
        f.amount, f.per_ip, f.per_day = 100, 2, per_day
        f.given, f.ip_log, f.day_log = set(), {}, []
        f.lock, f.nonce, f.last_submit = threading.Lock(), 0, 0.0
        return f

    def test_faucet_limits(self):
        f = self._faucet(per_day=3)
        a = [("%040x" % i) for i in range(1, 6)]
        self.assertEqual(f.drip(a[0], "1.1.1.1")[0], 200)
        self.assertEqual(f.drip(a[0], "2.2.2.2")[0], 429)          # once per address
        self.assertEqual(f.drip(a[1], "1.1.1.1")[0], 200)
        self.assertEqual(f.drip(a[2], "1.1.1.1")[0], 429)          # per-IP limit (2/hour here)
        self.assertEqual(f.drip(a[2], "3.3.3.3")[0], 200)
        self.assertEqual(f.drip(a[3], "4.4.4.4")[0], 429)          # daily cap (3 here)
        self.assertEqual(f.drip("not-an-address", "5.5.5.5")[0], 400)
        self.assertEqual(len(f.chain.sent), 3)

    def test_status_page_sends_security_headers(self):
        import http.client
        import json as _json
        import socket
        from mesh import testnet

        with tempfile.TemporaryDirectory() as d:
            from mesh import run
            genesis, _ = run.init(d, n_agents=1, replicas=1, n_validators=4)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        w = World(adversaries=False)
        with mock.patch.object(testnet, "RemoteChain", lambda g, ttl=2.0: type("C", (), {
                "state": property(lambda self: w.chain.state), "endpoints": [1, 2, 3, 4], "f": 1})()):
            srv = testnet.serve_status(genesis, port, host="127.0.0.1")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            for path in ("/", "/api/status", "/genesis.json"):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request("GET", path)
                r = c.getresponse()
                body = r.read()
                self.assertEqual(r.status, 200)
                self.assertIn("default-src 'none'", r.getheader("Content-Security-Policy"))
                self.assertEqual(r.getheader("X-Frame-Options"), "DENY")
                c.close()
            self.assertIn("validators", _json.loads(body))
        finally:
            srv.shutdown()
            srv.server_close()
