"""The playground: a web page where anyone can use the shared model.

Draw a digit (or, for the small "ring" demo task, click a point) and the page shows what the model thinks,
how sure it is, and which servers on the network computed each slice of the answer.

    python -m mesh.run app                     # a whole demo network in this one process, training as you watch
    python -m mesh.run node --name agent0 --ui-port 8000 ...   # an agent on a real network, with the page on

What the model is: a small neural network for recognising handwritten digits (MNIST), or points on a ring
in the quick demo task. It learns only from the training data its agents hold. It does not read websites,
and it is not a chatbot.
"""
import contextlib
import http.server
import json
import os
import threading
import time

import numpy as np

from .model import head
from .network import NodeDown, Rejected
from .webutil import _client_ip, _web_server

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MAX_BODY = 64 * 1024
PIXEL_MEAN, PIXEL_STD = 0.1307, 0.3081       # how mesh.datasets.mnist scales pixels
MAP_SIZE = 32                                # decision map resolution for the ring task
RING_RANGE = 5.0                             # the ring task's points lie within about +-5

APP_HEADERS = {
    # Scripts and styles only from this site, no inline code, no embedding in other sites.
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
}


# ---------------------------------------------------------------- turning a drawing into model input
def _resize(a, h, w, k=4):
    """Area-style resize: bilinear up to k times the target size, then average k x k blocks."""
    H, W = a.shape
    ys = np.clip((np.arange(h * k) + 0.5) * H / (h * k) - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w * k) + 0.5) * W / (w * k) - 0.5, 0, W - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    big = top * (1 - fy) + bot * fy
    return big.reshape(h, k, w, k).mean(axis=(1, 3))


def prepare_digit(img):
    """A square grayscale drawing (ink = 1, paper = 0) -> a 784-vector prepared the way MNIST digits are:
    the digit is cropped, scaled to fit a 20x20 box, and centred by its centre of mass in a 28x28 image."""
    img = np.asarray(img, float)
    if img.ndim != 2 or img.shape[0] != img.shape[1] or not 8 <= img.shape[0] <= 112:
        raise ValueError("expected a square image between 8x8 and 112x112")
    if not np.isfinite(img).all():
        raise ValueError("image has non-numbers in it")
    img = np.clip(img, 0, 1)
    if img.max() < 0.1:
        raise ValueError("the drawing is empty")
    rows, cols = np.where(img.max(1) > 0.05)[0], np.where(img.max(0) > 0.05)[0]
    crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.clip(_resize(crop, nh, nw), 0, 1)
    out = np.zeros((28, 28))
    top, left = (28 - nh) // 2, (28 - nw) // 2
    out[top:top + nh, left:left + nw] = small
    yy, xx = np.mgrid[0:28, 0:28]
    m = out.sum()
    dy, dx = int(round(14 - (yy * out).sum() / m)), int(round(14 - (xx * out).sum() / m))
    out = np.roll(out, (dy, dx), axis=(0, 1))     # the shift is small; the 20x20 box leaves a 4-pixel margin
    return ((out - PIXEL_MEAN) / PIXEL_STD).reshape(784)


def raw_pixels(x):
    """A prepared 784-vector back to 0..1 ink values, for showing a test digit on the page."""
    return np.clip(np.asarray(x) * PIXEL_STD + PIXEL_MEAN, 0, 1)


# ---------------------------------------------------------------- the model, as used by visitors
class VisitorsFirst:
    """The lock between the network's own work and visitors. The node holds it around each training step
    (`with guard:`) and visitors use `guard.visitor()`. A plain lock is not fair: a training loop that takes it
    again straight after letting go could keep visitors waiting for a whole round. Here the next training step
    waits while visitors are queued (for at most `max_wait_s`, so a crowd can't stall training)."""

    def __init__(self, max_wait_s=2.0):
        self.lock, self.cv, self.waiting, self.max_wait_s = threading.RLock(), threading.Condition(), 0, max_wait_s

    def __enter__(self):
        with self.cv:
            self.cv.wait_for(lambda: self.waiting == 0, timeout=self.max_wait_s)
        self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()

    @contextlib.contextmanager
    def visitor(self):
        with self.cv:
            self.waiting += 1
        try:
            self.lock.acquire()
        finally:
            with self.cv:
                self.waiting -= 1
                self.cv.notify_all()
        try:
            yield
        finally:
            self.lock.release()


class Playground:
    """Answers visitors' questions with one agent of the network. `guard` is a lock that the agent's own
    training loop also holds around each of its steps, so a visitor's request never lands mid-step."""

    def __init__(self, agent, chain, dataset, Xtest, ytest, mode, guard=None, target_rounds=None,
                 max_concurrent=4):
        self.agent, self.chain, self.dataset, self.mode = agent, chain, dataset, mode
        self.Xtest, self.ytest = Xtest, ytest
        self.guard = guard or VisitorsFirst()
        self.target_rounds = target_rounds
        self.history = []                    # [{"round", "acc"}] as training goes
        self.busy = threading.BoundedSemaphore(max_concurrent)
        self.rng = np.random.default_rng()
        self.served = 0

    @property
    def spec(self):
        return self.agent.spec

    def _names(self):
        return {a: n.get("name") or a[:8] for a, n in self.chain.state.nodes.items()}

    def _run(self, X):
        """Forward X through the network. Retries once: a round may switch while we ask."""
        last = None
        for _ in range(2):
            with self._visit():
                trace = []
                try:
                    st = self.chain.state
                    hs, _, _, ends = self.agent.forward(X, self.agent.versions(), private=False, trace=trace)
                    return head(ends, hs[-1]), trace, st.round
                except (NodeDown, Rejected) as e:
                    last = e
            time.sleep(0.3)
        raise NodeDown(str(last))

    def _visit(self):
        return self.guard.visitor() if hasattr(self.guard, "visitor") else self.guard

    def predict(self, x):
        probs, trace, rnd = self._run(np.asarray(x, float)[None, :])
        names = self._names()
        p = probs[0]
        self.served += 1
        return {
            "label": int(p.argmax()),
            "probs": [round(float(v), 4) for v in p],
            "route": [dict(t, name=names.get(t["server"], t["server"][:8]), server=t["server"][:8]) for t in trace],
            "agent": self.agent.name,
            "round": rnd,
            "model_version": self.agent.model_version()[:12],
        }

    def decision_map(self):
        """Ring task only: the predicted class over a grid of points."""
        g = np.linspace(-RING_RANGE, RING_RANGE, MAP_SIZE)
        X = np.array([[x, -y] for y in g for x in g])      # row 0 is the top of the picture (y = +5)
        probs, _, rnd = self._run(X)
        return {"size": MAP_SIZE, "range": RING_RANGE, "classes": probs.argmax(1).tolist(), "round": rnd}

    def sample(self):
        i = int(self.rng.integers(len(self.Xtest)))
        x, y = self.Xtest[i], int(self.ytest[i])
        if self.spec.d_in == 784:
            return {"pixels": [round(float(v), 3) for v in raw_pixels(x)], "label": y}
        return {"point": [round(float(v), 3) for v in x], "label": y}

    def info(self):
        return {"dataset": self.dataset, "d_in": self.spec.d_in, "classes": self.spec.n_classes, "mode": self.mode,
                "slices": self.spec.n_slices, "agent": self.agent.name}

    def status(self):
        st = self.chain.state
        hist = st.history[-20:]
        nodes = [{"name": n.get("name") or a[:8], "role": n["role"], "piece": n["piece"], "rep": n["rep"],
                  "active": n["active"]} for a, n in sorted(st.nodes.items(), key=lambda kv: (kv[1]["piece"],
                                                                                             kv[1].get("name", "")))]
        return {
            "mode": self.mode, "round": st.round, "height": st.height, "phase": st.phase,
            "target_rounds": self.target_rounds, "accuracy": self.history[-60:],
            "accepted_recent": sum(h["accepted"] for h in hist), "rounds_recent": len(hist),
            "validators": len(st.validators) or len(getattr(self.chain, "validators", [])), "nodes": nodes, "served": self.served,
            "events": st.events[-8:][::-1],
        }


# ---------------------------------------------------------------- the web server
class _Bucket:
    """Per-visitor rate limit for the model endpoints (each answer costs the network real work)."""

    def __init__(self, rate=2.0, burst=12):
        self.rate, self.burst, self.b, self.lock = rate, burst, {}, threading.Lock()

    def allow(self, ip):
        now = time.monotonic()
        with self.lock:
            tokens, last = self.b.get(ip, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self.b[ip] = (tokens - 1 if ok else tokens, now)
            if len(self.b) > 10_000:
                self.b = {k: v for k, v in self.b.items() if now - v[1] < 60}
            return ok


def _static():
    files = {}
    for name, ctype in (("index.html", "text/html; charset=utf-8"), ("app.js", "text/javascript; charset=utf-8"),
                        ("app.css", "text/css; charset=utf-8"), ("favicon.svg", "image/svg+xml")):
        with open(os.path.join(WEB_DIR, name), "rb") as f:
            files["/" + ("" if name == "index.html" else name)] = (f.read(), ctype)
    return files


def serve_app(pg, port, host="0.0.0.0", trust_proxy=False, rate=2.0, burst=12):
    """Serve the playground for `pg` on host:port in background threads. Returns the server."""
    static, bucket = _static(), _Bucket(rate, burst)

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", cache="no-store"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            for k, v in APP_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _model(self, fn, *args):
            if not bucket.allow(_client_ip(self, trust_proxy)):
                return self._send(429, {"error": "slow down a little: a few requests a second at most"})
            if not pg.busy.acquire(timeout=5):
                return self._send(503, {"error": "the playground is busy, try again in a moment"})
            try:
                return self._send(200, fn(*args))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except NodeDown:
                return self._send(503, {"error": "the network is switching to new weights, try again in a moment"})
            finally:
                pg.busy.release()

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in static:
                body, ctype = static[path]
                return self._send(200, body, ctype, cache="no-cache")
            if path == "/api/info":
                return self._send(200, pg.info())
            if path == "/api/status":
                return self._send(200, pg.status())
            if path == "/api/sample":
                return self._model(pg.sample)
            if path == "/api/map":
                if pg.spec.d_in != 2:
                    return self._send(404, {"error": "only for the ring task"})
                return self._model(pg.decision_map)
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/api/predict":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or -1)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self.close_connection = True
                return self._send(413, {"error": "request too large"})
            try:
                req = json.loads(self.rfile.read(n))
                x = _parse_input(req, pg.spec.d_in)
            except (ValueError, TypeError, KeyError) as e:
                return self._send(400, {"error": str(e) if isinstance(e, ValueError) else "bad request"})
            return self._model(pg.predict, x)

    srv = _web_server(host, port, H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _parse_input(req, d_in):
    if not isinstance(req, dict):
        raise ValueError("expected a JSON object")
    if d_in == 784:
        img = req.get("image")
        if not isinstance(img, list) or not img or not all(isinstance(v, (int, float)) for v in img):
            raise ValueError("send the drawing as {\"image\": [numbers between 0 and 1]}")
        side = int(round(len(img) ** 0.5))
        if side * side != len(img):
            raise ValueError("the image must be square")
        return prepare_digit(np.array(img, float).reshape(side, side))
    pt = req.get("point")
    if not isinstance(pt, list) or len(pt) != d_in or not all(isinstance(v, (int, float)) for v in pt):
        raise ValueError(f"send a point as {{\"point\": [{d_in} numbers]}}")
    x = np.array(pt, float)
    if not np.isfinite(x).all() or np.abs(x).max() > 100:
        raise ValueError("point out of range")
    return x


# ---------------------------------------------------------------- a whole demo network in one process
def run_demo(port=8000, host="0.0.0.0", dataset="mnist", rounds=150, attackers=True, trust_proxy=False,
             seed=0, stop=None):
    """Build a demo network (validators, 9 slice servers, 5 agents, and a few attackers if asked), serve the
    playground, and keep training in the background so visitors can watch the model learn."""
    from .sim import World
    print(f"building a demo network for the {dataset} task...", flush=True)
    w = World(dataset=dataset, adversaries=attackers, lr_decay_rounds=rounds, seed=seed)
    lock = VisitorsFirst()
    pg = Playground(w.agents[0], w.chain, dataset, w.Xtest, w.ytest, "demo", guard=lock, target_rounds=rounds)
    stop = stop or threading.Event()

    def evaluate():
        with lock:
            pg.history.append({"round": w.chain.state.round, "acc": round(w.accuracy(1000), 4)})

    def train():
        evaluate()
        while not stop.is_set() and (rounds is None or w.chain.state.round < rounds):
            w.round(guard=lock)
            if w.chain.state.round % 2 == 0:
                evaluate()
        if not pg.history or pg.history[-1]["round"] != w.chain.state.round:
            evaluate()
        print(f"training finished at round {w.chain.state.round}: "
              f"{pg.history[-1]['acc']:.1%} of test digits right", flush=True)

    t = threading.Thread(target=train, daemon=True)
    t.start()
    srv = serve_app(pg, port, host, trust_proxy=trust_proxy)
    print(f"playground on http://localhost:{port}  (training {rounds} rounds in the background)", flush=True)
    return pg, srv, t, stop

