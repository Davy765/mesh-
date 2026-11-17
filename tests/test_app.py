"""The playground: the web page anyone can use the model from."""
import http.client
import json
import os
import socket
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh import app  # noqa: E402
from mesh.sim import World  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    c.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        out = json.loads(raw)
    except ValueError:
        out = raw
    return r.status, out, r


class PrepareDigit(unittest.TestCase):
    def test_a_drawing_is_cropped_scaled_and_centred_like_mnist(self):
        img = np.zeros((56, 56))
        img[5:20, 30:50] = 1.0                       # a blob in the top-right corner
        x = app.raw_pixels(app.prepare_digit(img)).reshape(28, 28)
        rows, cols = np.where(x.max(1) > 0.05)[0], np.where(x.max(0) > 0.05)[0]
        self.assertLessEqual(max(len(rows), len(cols)), 22)       # fits the 20x20 box (plus resampling edge)
        yy, xx = np.mgrid[0:28, 0:28]
        self.assertAlmostEqual((yy * x).sum() / x.sum(), 14, delta=1.0)
        self.assertAlmostEqual((xx * x).sum() / x.sum(), 14, delta=1.0)

    def test_bad_drawings_are_refused(self):
        for bad in (np.zeros((56, 56)), np.ones((10, 20)), np.ones((4, 4)), np.full((28, 28), np.nan), np.ones(784)):
            with self.assertRaises(ValueError):
                app.prepare_digit(bad)

    @unittest.skipUnless(os.path.exists(os.path.expanduser("~/.cache/mesh/mnist/t10k-images-idx3-ubyte.gz")),
                         "MNIST not downloaded")
    def test_mnist_digits_come_back_unchanged(self):
        from mesh.datasets import mnist
        _, _, Xt, _ = mnist()
        for x in Xt[:100]:                         # an MNIST digit drawn on the pad is the digit itself again
            img = np.kron(app.raw_pixels(x).reshape(28, 28), np.ones((2, 2)))
            self.assertGreater(np.corrcoef(app.prepare_digit(img), x)[0, 1], 0.98)


class Playground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World(adversaries=True)            # the quick "ring" task: points in 2-D, 4 classes
        for _ in range(3):
            cls.w.round()
        cls.lock = app.VisitorsFirst()
        cls.pg = app.Playground(cls.w.agents[0], cls.w.chain, "ring", cls.w.Xtest, cls.w.ytest, "demo",
                                guard=cls.lock, target_rounds=10)
        cls.port = free_port()
        cls.srv = app.serve_app(cls.pg, cls.port, host="127.0.0.1", trust_proxy=True, rate=2, burst=8)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_pages_and_security_headers(self):
        for path, ctype in (("/", "text/html"), ("/app.js", "text/javascript"), ("/app.css", "text/css"),
                            ("/favicon.svg", "image/svg+xml"), ("/api/info", "application/json"),
                            ("/api/status", "application/json")):
            code, _, r = call(self.port, "GET", path)
            self.assertEqual(code, 200, path)
            self.assertTrue(r.getheader("Content-Type").startswith(ctype), path)
            csp = r.getheader("Content-Security-Policy")
            self.assertIn("script-src 'self'", csp)
            self.assertNotIn("unsafe-inline", csp)
            self.assertEqual(r.getheader("X-Frame-Options"), "DENY")
        self.assertEqual(call(self.port, "GET", "/nope")[0], 404)
        self.assertEqual(call(self.port, "GET", "/../mesh/app.py")[0], 404)
        _, page, _ = call(self.port, "GET", "/")
        self.assertNotIn(b"<script>", page)                 # no inline scripts anywhere
        self.assertNotIn(b" style=", page)

    def test_a_visitor_gets_an_answer_and_the_route_it_took(self):
        code, out, _ = call(self.port, "POST", "/api/predict", {"point": [3.0, 0.0]})
        self.assertEqual(code, 200, out)
        self.assertEqual(len(out["probs"]), 4)
        self.assertAlmostEqual(sum(out["probs"]), 1, places=2)
        self.assertEqual(out["label"], int(np.argmax(out["probs"])))
        self.assertEqual([h["piece"] for h in out["route"]], ["slice0", "slice1", "slice2"])
        st = self.w.chain.state
        for h in out["route"]:                       # every hop was answered by a real holder of that slice
            holders = [st.nodes[a]["name"] for a in st.members(h["piece"])]
            self.assertIn(h["name"], holders)

    def test_status_sample_and_map(self):
        code, s, _ = call(self.port, "GET", "/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(s["validators"], 4)
        self.assertEqual(s["target_rounds"], 10)
        self.assertTrue(any(n["name"].startswith("agent0") for n in s["nodes"]))
        code, smp, _ = call(self.port, "GET", "/api/sample")
        self.assertEqual(code, 200)
        self.assertEqual(len(smp["point"]), 2)
        code, m, _ = call(self.port, "GET", "/api/map")
        self.assertEqual(code, 200)
        self.assertEqual(len(m["classes"]), app.MAP_SIZE ** 2)
        self.assertTrue(set(m["classes"]) <= {0, 1, 2, 3})

    def test_bad_requests_are_refused_cleanly(self):
        for body in ({}, {"point": [1]}, {"point": ["a", "b"]}, {"point": [1e9, 0]}, [], "x", {"image": [1] * 784},
                     b"not json", b"\xff\xfe"):
            code, out, _ = call(self.port, "POST", "/api/predict", body)
            self.assertEqual(code, 400, (body, out))
            self.assertIn("error", out)
        code, _, _ = call(self.port, "POST", "/api/predict", b"x" * (app.MAX_BODY + 1))
        self.assertEqual(code, 413)
        self.assertEqual(call(self.port, "POST", "/api/other", {"point": [0, 0]})[0], 404)

    def test_each_visitor_is_rate_limited(self):
        codes = [call(self.port, "POST", "/api/predict", {"point": [0.5, 0.5]},
                      headers={"X-Forwarded-For": "198.51.100.7"})[0] for _ in range(20)]
        self.assertIn(429, codes)
        self.assertEqual(call(self.port, "POST", "/api/predict", {"point": [0.5, 0.5]},
                              headers={"X-Forwarded-For": "198.51.100.8"})[0], 200)   # others are unaffected


class DigitMode(unittest.TestCase):
    """The drawing-pad path, with a stand-in for the network (MNIST is not downloaded in CI)."""

    def test_drawings_are_prepared_and_answered(self):
        from mesh.model import Spec
        seen = []

        class FakeAgent:
            name, spec = "agent0", Spec(d_in=784, hidden=8, n_classes=10)

            def versions(self):
                return {}

            def model_version(self):
                return "f" * 64

            def forward(self, X, versions, private=True, trace=None):
                seen.append(X)
                trace.append({"piece": "slice0", "server": "a" * 40, "ms": 1.0, "failovers": 0,
                              "spot_checked": False, "caught_liar": False})
                p = np.full((len(X), 10), -1.0)
                p[:, 7] = 5
                return [p], None, None, {"W_out": np.eye(10), "b_out": np.zeros(10)}

        class FakeChain:
            class state:
                round, nodes = 3, {}

        pg = app.Playground(FakeAgent(), FakeChain(), "mnist", np.zeros((5, 784)), np.zeros(5, int), "network")
        import mesh.app as appmod
        old = appmod.head
        appmod.head = lambda ends, h: np.exp(h) / np.exp(h).sum(1, keepdims=True)
        port = free_port()
        srv = app.serve_app(pg, port, host="127.0.0.1")
        try:
            img = np.zeros((56, 56))
            img[10:46, 26:30] = 1                    # a "1"
            code, out, _ = call(port, "POST", "/api/predict", {"image": img.ravel().tolist()})
            self.assertEqual(code, 200, out)
            self.assertEqual(out["label"], 7)
            self.assertEqual(seen[-1].shape, (1, 784))
            code, out, _ = call(port, "POST", "/api/predict", {"image": [0] * 3136})
            self.assertEqual(code, 400)
            self.assertIn("empty", out["error"])
            code, smp, _ = call(port, "GET", "/api/sample")
            self.assertEqual(len(smp["pixels"]), 784)
        finally:
            appmod.head = old
            srv.shutdown()
            srv.server_close()


class VisitorsGoFirst(unittest.TestCase):
    def test_a_busy_training_loop_does_not_keep_visitors_waiting(self):
        guard, stop = app.VisitorsFirst(), threading.Event()

        def trainer():                           # takes the lock again the moment it lets go, like a node does
            while not stop.is_set():
                with guard:
                    time.sleep(0.05)
        t = threading.Thread(target=trainer)
        t.start()
        try:
            waits = []
            for _ in range(10):
                t0 = time.perf_counter()
                with guard.visitor():
                    waits.append(time.perf_counter() - t0)
                time.sleep(0.02)
        finally:
            stop.set()
            t.join()
        self.assertLess(max(waits), 0.5)        # at most about one training step


class VisitorsDuringTraining(unittest.TestCase):
    def test_visitors_can_use_the_model_while_it_trains(self):
        w = World(adversaries=True)
        lock = app.VisitorsFirst()
        pg = app.Playground(w.agents[0], w.chain, "ring", w.Xtest, w.ytest, "demo", guard=lock)
        stop, results = threading.Event(), []

        def visitor():
            rng = np.random.default_rng(0)
            while not stop.is_set():
                try:
                    results.append(pg.predict(rng.uniform(-4, 4, 2))["round"])
                except app.NodeDown:
                    results.append(None)
                time.sleep(0.01)
        r0 = w.chain.state.round
        threads = [threading.Thread(target=visitor) for _ in range(3)]
        for t in threads:
            t.start()
        try:
            for _ in range(8):
                w.round(guard=lock)
        finally:
            stop.set()
            for t in threads:
                t.join()
        st = w.chain.state
        self.assertEqual(st.round, r0 + 8)
        self.assertGreaterEqual(sum(h["accepted"] for h in st.history), 5)    # training was not disturbed
        answered = [r for r in results if r is not None]
        self.assertGreater(len(answered), 20)
        self.assertGreaterEqual(len(answered) / len(results), 0.95)
        self.assertGreater(len(set(answered)), 3)              # visitors saw the model change as it trained


if __name__ == "__main__":
    unittest.main()
