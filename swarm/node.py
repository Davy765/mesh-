"""A swarm peer: trains on its own data and gossips weights with other peers.

There is no coordinator. Each node runs an HTTP server and a background loop:

    repeat:
        train one epoch on local (private) data
        pick a random known peer
        push-pull: send my weights, receive theirs, both sides average
        merge peer lists (membership gossip)

Raw training data never leaves a node; only model weights do.
"""
import hashlib
import hmac
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .model import MLP

MAX_BODY = 10 * 1024 * 1024
MAX_PEERS = 64


class Node:
    def __init__(self, X, y, *, host="127.0.0.1", port=0, peers=(), model_shape=(2, 32, 4),
                 model_seed=0, secret=None, lr=0.1, round_interval=0.05, max_drift=None,
                 node_id=None):
        self.id = node_id or uuid.uuid4().hex[:8]
        self.X, self.y = np.asarray(X, float), np.asarray(y, int)
        self.model = MLP(*model_shape, seed=model_seed)
        self.lr = lr
        self.round_interval = round_interval
        self.max_drift = max_drift          # clip how far one peer can pull us (poison guard)
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.lock = threading.Lock()
        self.rounds = 0
        self.rejected = 0
        self.peers = set(peers)
        self.failures = {}
        self._rng = np.random.default_rng()
        self._stop = threading.Event()

        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.address = f"http://{host}:{self.server.server_address[1]}"
        self._threads = []

    # ------------------------------------------------------------------ life
    def start(self, gossip=True):
        t = threading.Thread(target=self.server.serve_forever, daemon=True)
        t.start()
        self._threads.append(t)
        if gossip:
            g = threading.Thread(target=self._loop, daemon=True)
            g.start()
            self._threads.append(g)
        return self

    def stop(self):
        self._stop.set()
        self.server.shutdown()
        self.server.server_close()

    # ----------------------------------------------------------------- learn
    def _loop(self):
        while not self._stop.is_set():
            self.step()
            self._stop.wait(self.round_interval)

    def step(self):
        """One round: local training, then one gossip exchange."""
        with self.lock:
            if len(self.X):
                self.model.train(self.X, self.y, epochs=1, lr=self.lr, rng=self._rng)
            self.rounds += 1
        peer = self._pick_peer()
        if peer:
            self.gossip_with(peer)

    def _pick_peer(self):
        candidates = sorted(self.peers - {self.address})
        return random.choice(candidates) if candidates else None

    def gossip_with(self, peer):
        with self.lock:
            payload = self._payload()
        try:
            reply = self._post(peer + "/gossip", payload)
        except (urllib.error.URLError, OSError, ValueError):
            self.failures[peer] = self.failures.get(peer, 0) + 1
            if self.failures[peer] >= 3:          # forget dead peers
                self.peers.discard(peer)
                self.failures.pop(peer, None)
            return False
        self.failures.pop(peer, None)
        self._absorb(reply)
        return True

    def _payload(self):
        return {"id": self.id, "address": self.address, "weights": self.model.get_weights(),
                "samples": int(len(self.X)), "peers": sorted(self.peers)[:MAX_PEERS]}

    def _absorb(self, msg):
        """Average a peer's weights into ours, after sanity checks."""
        for p in msg.get("peers", []) + [msg.get("address")]:
            if isinstance(p, str) and p.startswith("http") and p != self.address \
                    and len(self.peers) < MAX_PEERS:
                self.peers.add(p)
        try:
            theirs = {k: np.asarray(v, float) for k, v in msg["weights"].items()}
        except (KeyError, TypeError, ValueError, AttributeError):
            self.rejected += 1
            return False
        with self.lock:
            mine = self.model.params
            if theirs.keys() != mine.keys() or any(theirs[k].shape != mine[k].shape for k in mine) \
                    or not all(np.isfinite(v).all() for v in theirs.values()):
                self.rejected += 1
                return False
            delta = {k: theirs[k] - mine[k] for k in mine}
            if self.max_drift is not None:
                norm = np.sqrt(sum((d ** 2).sum() for d in delta.values()))
                if norm > self.max_drift:
                    delta = {k: d * (self.max_drift / norm) for k, d in delta.items()}
            for k in mine:
                mine[k] = mine[k] + 0.5 * delta[k]     # pairwise average
        return True

    # ------------------------------------------------------------------ http
    def _sign(self, body):
        return hmac.new(self.secret, body, hashlib.sha256).hexdigest() if self.secret else ""

    def _verify(self, body, sig):
        return not self.secret or hmac.compare_digest(self._sign(body), sig or "")

    def _post(self, url, obj):
        body = json.dumps(obj).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "X-Swarm-Sig": self._sign(body)})
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read(MAX_BODY)
            if not self._verify(raw, r.headers.get("X-Swarm-Sig")):
                raise ValueError("bad signature on reply")
            return json.loads(raw)

    def _handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Swarm-Sig", node._sign(body))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/info":
                    self._send(200, {"id": node.id, "address": node.address, "rounds": node.rounds,
                                     "samples": int(len(node.X)), "peers": sorted(node.peers),
                                     "rejected": node.rejected})
                elif self.path == "/model":
                    with node.lock:
                        self._send(200, {"weights": node.model.get_weights()})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    return self._send(413, {"error": "too large"})
                body = self.rfile.read(length)
                if not node._verify(body, self.headers.get("X-Swarm-Sig")):
                    return self._send(401, {"error": "bad signature"})
                try:
                    msg = json.loads(body)
                except ValueError:
                    return self._send(400, {"error": "bad json"})
                if self.path == "/gossip":
                    with node.lock:
                        reply = node._payload()        # reply with pre-merge weights
                    node._absorb(msg)
                    self._send(200, reply)
                elif self.path == "/predict":
                    try:
                        X = np.asarray(msg["inputs"], float)
                        with node.lock:
                            probs = node.model.predict_proba(X)
                    except (KeyError, ValueError):
                        return self._send(400, {"error": "expected {'inputs': [[...], ...]}"})
                    self._send(200, {"predictions": probs.argmax(1).tolist(),
                                     "probabilities": probs.round(4).tolist()})
                else:
                    self._send(404, {"error": "not found"})

        return Handler
