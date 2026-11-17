"""An agent: owns private data, holds the model's ends, uses (and pays for) the slice servers,
trains the shared model, votes on updates, and trades work with other agents.
"""
import time

import numpy as np

from .aggregate import clip, cosine
from .crypto import canon, merkle_root, sha, verify_obj
from .member import PieceMember
from .model import embed, embed_grads, head, loss_and_head_grads, output_hash
from .network import NodeDown, Rejected, check_array
from .server import input_fingerprint


def pipeline_makespan(times):
    """times[m][s]: time of micro-batch m at stage s. Stages work on different micro-batches at once."""
    M, S = len(times), len(times[0])
    fin = [[0.0] * S for _ in range(M)]
    for m in range(M):
        for s in range(S):
            fin[m][s] = max(fin[m - 1][s] if m else 0, fin[m][s - 1] if s else 0) + times[m][s]
    return fin[-1][-1]


class Agent(PieceMember):
    def __init__(self, identity, name, spec, chain, X, y, *, spot_check=0.15, activation_noise=0.0,
                 batch=64, val_frac=0.2, max_val=500, seed_set=None, **kw):
        super().__init__(identity, name, "ends", spec, chain, **kw)
        n_val = min(int(len(X) * val_frac), max_val)   # held-out check set, kept small: it crosses the network
        self.Xval, self.yval, self.Xtr, self.ytr = X[:n_val], y[:n_val], X[n_val:], y[n_val:]
        self.spot_check, self.activation_noise, self.batch = spot_check, activation_noise, batch
        self.seed_set = seed_set           # small public labelled set, fixed at genesis (or None)
        self.ewma, self.blocked = {}, set()
        self.work, self.receipts = {}, []
        self.stats = {"hops_fast": 0, "ms_fast": 0.0, "hops_spread": 0, "ms_spread": 0.0,
                      "failovers": 0, "checks": 0, "caught": 0}
        self.my_jobs, self.job_results = {}, {}
        self.withhold_to = set()          # attack demo: the replicas a "withholder" agent trains through
        self.last_loss = None

    # ---- routing -------------------------------------------------------------------
    def choose(self, piece, exclude=(), route="fast"):
        """Pick a replica. Servers with low on-chain reputation (i.e. slashed) are avoided.
        route="fast":   fastest measured round-trip first (untried servers get tried once) - for inference
        route="spread": spread evenly - for training, so several replicas contribute updates and
                        poison-resistant aggregation has something to compare."""
        st = self.chain.state
        cands = [a for a in st.members(piece) if a not in exclude and a not in self.blocked]
        if not cands:
            raise NodeDown(f"no replicas left for {piece}")
        good = [a for a in cands if st.nodes[a]["rep"] >= 5] or cands
        if route == "spread":
            return good[int(self.rng.integers(len(good)))]
        score = np.array([self.ewma.get(a, 0.0) for a in good])
        w = np.exp(-(score - score.min()) / 10.0)
        return good[int(self.rng.choice(len(good), p=w / w.sum()))]

    def _call(self, addr, kind, version, arrays, train=False):
        req = {"version": version, "train": train}
        if train:
            req["round"] = self.train_round            # which round this training step is for
        meta, out, ms = self.net.call(self, addr, kind, req, arrays)
        self.ewma[addr] = 0.7 * self.ewma.get(addr, ms) + 0.3 * ms
        rc, node = meta.get("receipt", {}), self.chain.state.nodes[addr]
        b = rc.get("body", {})
        want_in = output_hash(arrays["h"]) if kind == "fwd" else sha(output_hash(arrays["h"]), output_hash(arrays["g"]))
        if b.get("by") != addr or b.get("version") != version or b.get("mode") != kind or b.get("input") != want_in \
                or b.get("output") != output_hash(out["out"]) or not verify_obj(node["sign_pk"], rc.get("sig", ""), b):
            raise Rejected("bad receipt")
        return out["out"], rc, ms

    def _pay(self, addr, rc, units=1):
        self.work[addr] = self.work.get(addr, 0) + units
        self.receipts.append(sha(canon(rc)))

    def run_hop(self, piece, kind, version, arrays, train=False, route="fast"):
        """One hop through a slice, with failover and random spot checks. Returns (output, ms)."""
        ms_total, tried, sweeps, why = 0.0, set(), 0, {}
        for _ in range(24):                        # enough for every replica on every retry below
            try:
                addr = self.choose(piece, exclude=tried, route=route)
            except NodeDown as e:
                if sweeps >= 3 or not tried:       # retried enough, or nobody to ask at all
                    reasons = "; ".join(f"{a[:8]}: {r}" for a, r in why.items())
                    raise NodeDown(f"{e}" + (f" ({reasons})" if reasons else "")) from None
                wait = 0.25 * 2 ** sweeps          # every replica refused: maybe they're still switching to
                sweeps += 1                        # this round's weights (slower on a busy machine). Wait a
                time.sleep(wait)                   # little longer each time: 0.25 s, 0.5 s, 1 s.
                ms_total += wait * 1000
                tried = set()
                continue
            tried.add(addr)
            try:
                out, rc, ms = self._call(addr, kind, version, arrays, train)
            except NodeDown:
                ms_total += self.net.timeout_ms
                self.ewma[addr] = self.net.timeout_ms
                self.stats["failovers"] += 1
                why[addr] = "down"
                continue
            except Rejected as e:
                ms_total += 2 * self.latency_ms
                self.stats["failovers"] += 1
                why[addr] = str(e)
                continue
            ms_total += ms
            self._pay(addr, rc)
            if self.rng.random() < self.spot_check:
                out = self.spot_check_hop(piece, kind, version, arrays, addr, out, rc)
            self.stats[f"hops_{route}"] += 1
            self.stats[f"ms_{route}"] += ms
            return out, ms_total
        raise NodeDown(f"no working replica for {piece}")  # pragma: no cover

    def spot_check_hop(self, piece, kind, version, arrays, addr, out, rc):
        """Ask other replicas to redo the same computation. If two of them agree on a different
        answer, the signed receipts prove who lied, and anyone can check that on chain."""
        self.stats["checks"] += 1
        others, exclude = [], {addr}
        while len(others) < 2:
            try:
                a2 = self.choose(piece, exclude=exclude)
            except NodeDown:
                break
            exclude.add(a2)
            try:
                o2, r2, _ = self._call(a2, kind, version, arrays, train=False)
            except (NodeDown, Rejected):
                continue
            self._pay(a2, r2)
            others.append((a2, o2, r2))
            if len(others) == 1 and r2["body"]["output"] == rc["body"]["output"]:
                return out                                   # agreed: fine
        if len(others) < 2:
            return out
        (a2, o2, r2), (a3, o3, r3) = others
        h1, h2, h3 = rc["body"]["output"], r2["body"]["output"], r3["body"]["output"]
        if h2 == h3 != h1:
            accused, ev, fixed = addr, {"accused": rc, "witnesses": [r2, r3]}, o2
        elif h1 == h3 != h2:
            accused, ev, fixed = a2, {"accused": r2, "witnesses": [rc, r3]}, out
        else:
            return out
        self.blocked.add(accused)
        self.work[accused] -= 1                             # don't pay for the wrong answer
        self.stats["caught"] += 1
        self.submit("slash", {"evidence": ev})
        return fixed

    # ---- running the whole model ----------------------------------------------------------
    def versions(self, which="current"):
        st = self.chain.state
        if which == "current":
            return {p: st.canonical[p] for p in self.spec.pieces}
        return {p: st.cur["chosen"][p][0] for p in self.spec.pieces}

    def model_version(self):
        return sha(canon(self.versions("current")))

    def forward(self, X, versions, route="fast", private=True):
        ends = self.params_for(versions["ends"])
        h0, z0 = embed(ends, X)
        if self.activation_noise and private:      # servers only ever see noised hidden features
            h0 = h0 + self.rng.normal(0, self.activation_noise, h0.shape)
        hs, hop_ms = [h0], []
        for i in range(self.spec.n_slices):
            out, ms = self.run_hop(f"slice{i}", "fwd", versions[f"slice{i}"], {"h": hs[-1]}, route=route)
            hs.append(out)
            hop_ms.append(ms)
        return hs, z0, hop_ms, ends

    def predict(self, X, versions=None):
        hs, _, _, ends = self.forward(X, versions or self.versions())
        return head(ends, hs[-1]).argmax(1)

    def infer(self, X, microbatches=1):
        """Inference with micro-batches flowing through the slice pipeline at the same time."""
        labels, times = [], []
        for part in np.array_split(X, microbatches):
            hs, _, hop_ms, ends = self.forward(part, self.versions())
            labels.append(head(ends, hs[-1]).argmax(1))
            times.append(hop_ms)
        return np.concatenate(labels), {"sequential_ms": float(np.sum(times)),
                                        "pipelined_ms": pipeline_makespan(times)}

    def val_loss(self, versions):
        hs, _, _, ends = self.forward(self.Xval, versions)
        p = head(ends, hs[-1])
        return float(-np.log(p[np.arange(len(self.yval)), self.yval] + 1e-12).mean())

    # ---- training round -------------------------------------------------------------------
    def train_step(self):
        if not self.active or not self.online:
            return
        v = self.versions()
        idx = self.rng.choice(len(self.Xtr), self.batch)
        xb, yb = self.Xtr[idx], self.ytr[idx]
        hs, z0, _, ends = self.forward(xb, v, route="spread")
        self.last_loss, _, g_head, g = loss_and_head_grads(ends, hs[-1], yb)
        self.round_inputs = {}
        self.train_round = self.chain.state.round
        for i in reversed(range(self.spec.n_slices)):
            g = self.train_hop(f"slice{i}", v[f"slice{i}"], hs[i], g)
        self.add_contribution(clip({**g_head, **embed_grads(xb, z0, g)}, self.C))
        self.submit("contrib", {"round": self.chain.state.round,
                                "inputs": {p: input_fingerprint(h, gg) for p, (h, gg) in self.round_inputs.items()}})

    def train_hop(self, piece, version, h, g):
        """Send this training step to EVERY replica of the slice. They all record my contribution (so they
        can all compute the same update) and all return the gradient, so every training hop is fully
        cross-checked: a replica that disagrees with two others is proven wrong on the spot."""
        self.round_inputs[piece] = (h, g)
        answers, slowest, why = [], 0.0, {}
        targets = self.chain.state.members(piece)
        if self.behaviour == "withholder":                 # attack: train through only some replicas
            targets = [t for t in targets if t in self.withhold_to] or targets[:1]
        pending = list(targets)
        for attempt in range(4):
            for addr in list(pending):
                try:
                    out, rc, ms = self._call(addr, "bwd", version, {"h": h, "g": g}, train=True)
                except (NodeDown, Rejected) as e:
                    self.stats["failovers"] += 1
                    why[addr] = str(e) if isinstance(e, Rejected) else "down"
                    continue
                pending.remove(addr)
                self._pay(addr, rc)
                answers.append((addr, out, rc))
                slowest = max(slowest, ms)                 # the calls could go out in parallel
            # Replicas that refused may still be switching to this round's weights (slow on a busy machine).
            # Give them a little more time: the update is only identical everywhere if they all get my input.
            if not pending or attempt == 3 or all(why.get(a) == "down" for a in pending):
                break
            time.sleep(0.25 * 2 ** attempt)
        if not answers:
            reasons = "; ".join(f"{a[:8]}: {r}" for a, r in why.items())
            raise NodeDown(f"no working replica for {piece} ({reasons})")
        groups = {}
        for a in answers:
            groups.setdefault(a[2]["body"]["output"], []).append(a)
        best = max(groups.values(), key=len)
        if len(best) >= 2:
            for addr, _, rc in answers:
                if rc["body"]["output"] != best[0][2]["body"]["output"] and addr not in self.blocked:
                    self.blocked.add(addr)
                    self.work[addr] -= 1
                    self.stats["caught"] += 1
                    self.submit("slash", {"evidence": {"accused": rc, "witnesses": [best[0][2], best[1][2]]}})
        self.stats["checks"] += len(answers) > 1
        self.stats["hops_spread"] += 1
        self.stats["ms_spread"] += slowest
        return best[0][1]

    def phase_vote(self):
        if not self.active or not self.online:
            return
        st = self.chain.state
        delta = 1.0                                       # "worse" unless I can show otherwise
        if all(p in st.cur["chosen"] for p in self.spec.pieces):
            cand = self.versions("candidate")
            if self.cand_hash != cand["ends"]:          # I computed something else: fetch the majority's
                p = self.fetch_verified(cand["ends"])
                if p is not None:
                    self.cand, self.cand_hash = p, cand["ends"]
            if self.cand_hash == cand["ends"]:
                try:
                    cur = self.val_loss(self.versions())
                    delta = (self.val_loss(cand) - cur) / max(cur, 1e-6)   # relative change: +0.3 = 30% worse
                except NodeDown:
                    pass
        if self.behaviour == "poisoner":
            delta = -1.0                                  # "it's great, ship it"
        self.submit("vote", {"round": st.round, "delta": round(delta, 6)})
        self.submit("settle", {"round": st.round, "pay": dict(self.work),
                               "receipts_root": merkle_root(self.receipts)})
        self.work, self.receipts = {}, []

    def trust_scores(self, updates):
        """How far each agent's update points the same way as a training step on the public seed set.
        Only the output layer is compared: it is where each class has its own weights, so an update that
        un-learns some classes points clearly against the seed-set step."""
        if self.seed_set is None:
            return None
        Xs, ys = self.seed_set
        try:
            # public data, so no privacy noise: every agent must get exactly the same trust scores
            hs, _, _, ends = self.forward(Xs, self.versions(), private=False)
        except NodeDown:
            return None
        _, _, g_head, _ = loss_and_head_grads(ends, hs[-1], ys)
        ref = -np.concatenate([g_head["W_out"].ravel(), g_head["b_out"].ravel()])
        # rounded so that agents whose forward passes went through different (honest) machines still agree
        return [round(max(0.0, cosine(np.concatenate([u["W_out"].ravel(), u["b_out"].ravel()]), ref)), 4)
                for u in updates]

    # ---- job market ------------------------------------------------------------------------
    def claim_receipt(self, job, chunk, input_hash, labels):
        body = {"by": self.address, "job": job, "chunk": chunk, "input": input_hash,
                "version": self.model_version(), "output": sha(np.asarray(labels, np.int64).tobytes())}
        return {"body": body, "sig": self.id.sign_obj(body)}

    def post_job(self, job, X, n_chunks, reward):
        self.my_jobs[job] = np.array_split(X, n_chunks)
        self.submit("job_open", {"job": job, "reward": reward,
                                 "inputs": [output_hash(c) for c in self.my_jobs[job]]})

    def claim(self, job, chunk):
        self.submit("job_claim", {"job": job, "chunk": chunk})

    def work_chunk(self, job, chunk, microbatches=4):
        j = self.chain.state.jobs[job]
        _, arrays, _ = self.net.call(self, j["poster"], "job_data", {"job": job, "chunk": chunk})
        X = check_array(arrays.get("x"), 2, cols=self.spec.d_in, max_rows=10_000)
        if output_hash(X) != j["chunks"][chunk]["input"]:
            raise Rejected("poster sent data that doesn't match the job")
        labels, timing = self.infer(X, microbatches)
        if self.behaviour == "poisoner":                  # lazy worker: makes answers up
            labels = self.rng.integers(0, self.spec.n_classes, len(labels))
        rc = self.claim_receipt(job, chunk, output_hash(X), labels)
        self.net.call(self, j["poster"], "job_result", {"job": job, "chunk": chunk, "receipt": rc},
                      {"labels": labels})
        self.submit("job_submit", {"job": job, "chunk": chunk, "result": rc["body"]["output"]})
        return timing

    def review(self, job, chunk, sample=6):
        """Spot-check a few answers myself; pay if right, otherwise prove it was wrong."""
        labels, rc = self.job_results[(job, chunk)]
        X = self.my_jobs[job][chunk]
        idx = self.rng.choice(len(X), min(sample, len(X)), replace=False)
        if (self.predict(X[idx]) == labels[idx]).all():
            self.submit("job_release", {"job": job, "chunk": chunk})
            return "paid"
        st, worker = self.chain.state, rc["body"]["by"]
        claims = []
        for a in st.members("ends"):
            if a in (self.address, worker):
                continue
            try:
                meta, _, _ = self.net.call(self, a, "job_attest", {"job": job, "chunk": chunk}, {"x": X})
                claims.append(meta["receipt"])
            except (NodeDown, Rejected):
                continue
        for i in range(len(claims)):
            for k in range(i + 1, len(claims)):
                if claims[i]["body"]["output"] == claims[k]["body"]["output"] != rc["body"]["output"]:
                    self.submit("job_dispute", {"job": job, "chunk": chunk,
                                                "evidence": {"accused": rc, "witnesses": [claims[i], claims[k]]}})
                    return "disputed"
        return "unresolved"

    def handle(self, kind, sender, meta, arrays):
        r = self.handle_member(kind, sender, meta, arrays)
        if r is not None:
            return r
        st = self.chain.state
        node = st.nodes.get(sender)
        if kind == "contrib_data" and node and node["active"] and node["role"] == "server":
            pass
        elif not node or not node["active"] or node["role"] != "client":
            raise Rejected("only registered agents")
        if kind == "contrib_data":            # a replica missed my training input: it can have it
            if self.behaviour == "withholder":
                raise Rejected("not giving it to you")
            if sender not in st.members(meta["piece"]) or meta["round"] != st.round \
                    or meta["piece"] not in getattr(self, "round_inputs", {}):
                raise Rejected("not yours")
            h, g = self.round_inputs[meta["piece"]]
            return {}, {"h": h, "g": g}, 1.0
        if kind == "job_data":
            j = st.jobs.get(meta["job"])
            if not j or j["poster"] != self.address or j["chunks"][meta["chunk"]]["worker"] != sender:
                raise Rejected("not your chunk")
            return {}, {"x": self.my_jobs[meta["job"]][meta["chunk"]]}, 1.0
        if kind == "job_result":
            rc, labels = meta["receipt"], arrays.get("labels")
            if not isinstance(labels, np.ndarray) or labels.ndim != 1 or labels.dtype.kind not in "iu":
                raise Rejected("malformed labels")
            if rc["body"]["by"] != sender or rc["body"]["output"] != sha(np.asarray(labels, np.int64).tobytes()) \
                    or not verify_obj(node["sign_pk"], rc["sig"], rc["body"]):
                raise Rejected("result does not match its receipt")
            self.job_results[(meta["job"], meta["chunk"])] = (labels, rc)
            return {}, {}, 1.0
        if kind == "job_attest":
            X = check_array(arrays.get("x"), 2, cols=self.spec.d_in, max_rows=10_000)
            labels, t = self.infer(X)
            if self.behaviour == "poisoner":
                labels = self.rng.integers(0, self.spec.n_classes, len(labels))
            return {"receipt": self.claim_receipt(meta["job"], meta["chunk"], output_hash(X), labels)}, {}, \
                t["sequential_ms"]
        raise Rejected(f"unknown request {kind}")
