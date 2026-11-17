"""A server: holds one slice of the model and computes it for registered agents, for pay.

Training is verifiable: every agent sends its training inputs for this slice to EVERY replica and declares
their fingerprint on chain. Each replica turns exactly the declared inputs into an update, so all honest
replicas compute bit-identical candidate weights and the chain keeps the majority's. A server that
computes anything else (lazy, buggy or poisoning) is simply outvoted, earns nothing, and has to resync.
"""
import os

import numpy as np

from .aggregate import clip, dp_release
from .crypto import sha
from .member import PieceMember
from .model import output_hash, param_hash, quantize, slice_backward, slice_forward
from .network import NodeDown, Rejected, check_array


def input_fingerprint(h, g):
    return sha(output_hash(h), output_hash(g))


class SliceServer(PieceMember):
    def __init__(self, *a, compute_ms_per_item=0.02, lie_rate=0.5, peer_relay=True, **kw):
        super().__init__(*a, **kw)
        self.peer_relay = peer_relay     # fetch inputs/salts I was left out of from other replicas
        self.quantize = True             # snap updates to a grid so different hardware still agrees
        self.float_jitter = 0.0          # tests only: imitate a machine whose arithmetic differs in the last bits
        self.compute_ms_per_item = compute_ms_per_item
        self.lie_rate = lie_rate
        self.served = 0

    def _reset_round(self):
        super()._reset_round()
        self.contribs = {}            # agent -> (input fingerprint, clipped gradient)
        self.salts, self.my_salt = {}, None

    def _jitter(self, a):
        if not self.float_jitter:
            return a
        return a * (1 + self.float_jitter * self.rng.standard_normal(np.shape(a)))

    def receipt(self, mode, version, input_hash, out):
        body = {"by": self.address, "piece": self.piece, "version": version, "mode": mode,
                "input": input_hash, "output": output_hash(out)}
        return {"body": body, "sig": self.id.sign_obj(body)}

    def _maybe_lie(self, out):
        if self.behaviour == "liar" and self.rng.random() < self.lie_rate:
            return out + self.rng.normal(0, 1.0, out.shape)     # skip the work / corrupt the answer
        return out

    # ---- training round ----------------------------------------------------------------------
    def phase_commit(self):
        """Commit to a random salt; the salts are combined into the seed for privacy noise, so the
        noise is identical on every honest replica but unknown to anyone outside the slice."""
        if not self.active or not self.online:
            return
        self.my_salt = os.urandom(16).hex()
        self.submit("commit", {"round": self.chain.state.round, "piece": self.piece, "digest": sha(self.my_salt)})

    def phase_reveal(self):
        if self.my_salt is None or not self.online:
            return
        with self.lock:
            self.salts[self.address] = self.my_salt
        peers = self.peers_of_piece()
        if self.behaviour == "withholder":                 # attack: tell only one peer
            peers = peers[:1]
        for peer in peers:
            try:
                self.net.call(self, peer, "salt", {"round": self.chain.state.round, "salt": self.my_salt})
            except (NodeDown, Rejected):
                pass

    def _contribution(self, agent, want):
        """The clipped gradient for one agent's declared input. If I missed the input, get it from the agent
        or, failing that, from any other replica of my slice. The on-chain fingerprint proves it is the real
        input, so the source doesn't need to be trusted, and an agent can't split the replicas by sending its
        input to only some of them: if any honest replica got it, all of them do."""
        with self.lock:
            have = self.contribs.get(agent)
        if have and have[0] == want:
            return have[1]
        rnd = self.chain.state.round
        sources = [(agent, "contrib_data", {"round": rnd, "piece": self.piece})]
        if self.peer_relay:
            sources += [(p, "get_contrib", {"round": rnd, "agent": agent}) for p in self.peers_of_piece()]
        for addr, kind, meta in sources:
            try:
                _, arrays, _ = self.net.call(self, addr, kind, meta)
            except (NodeDown, Rejected):
                continue
            try:                                           # a wrong-shaped input would crash every replica the
                h = check_array(arrays.get("h"), 2, cols=self.spec.hidden)   # same way: skip it instead
                g = check_array(arrays.get("g"), 2, cols=self.spec.hidden)
            except Rejected:
                continue
            if len(h) != len(g) or input_fingerprint(arrays["h"], arrays["g"]) != want:
                continue
            grads, _ = slice_backward(self.params, arrays["h"], arrays["g"])
            grads = {k: self._jitter(v) for k, v in grads.items()}
            with self.lock:
                self.contribs[agent] = (want, clip(grads, self.C), arrays["h"], arrays["g"])
            return self.contribs[agent][1]
        return None

    def _missing_salts(self, commits):
        """Get committed salts I was not sent from other replicas (checked against the on-chain hash)."""
        with self.lock:
            missing = [a for a in commits if a not in self.salts]
        for a in missing:
            for peer in self.peers_of_piece():
                if peer == a:
                    continue
                try:
                    meta, _, _ = self.net.call(self, peer, "get_salt", {"round": self.chain.state.round, "of": a})
                except (NodeDown, Rejected):
                    continue
                salt = str(meta.get("salt", ""))
                if sha(salt) == commits[a]:
                    with self.lock:
                        self.salts[a] = salt
                    break

    def phase_candidate(self):
        if not self.active or not self.online:
            return
        st = self.chain.state
        declared = st.cur.get("contribs", {}).get(self.piece, {})
        total, n = {k: np.zeros_like(v) for k, v in self.params.items()}, 0
        for agent in sorted(declared):                     # fixed order -> bit-identical sums everywhere
            g = self._contribution(agent, declared[agent])
            if g is None:
                continue
            for k in total:
                total[k] += g[k]
            n += 1
        if n == 0:
            return
        commits = st.cur["commits"].get(self.piece, {})
        if self.peer_relay and self.sigma > 0:
            self._missing_salts(commits)
        with self.lock:
            salts = sorted((a, s) for a, s in self.salts.items() if commits.get(a) == sha(s))
        seed = int(sha(st.round, self.piece, *[a + s for a, s in salts])[:16], 16)
        mean = dp_release(total, n, self.C, self.sigma, np.random.default_rng(seed))
        update = {k: -self.current_lr() * v for k, v in mean.items()}
        if self.behaviour == "poisoner":                   # push the model the wrong way, hard
            update = {k: -50 * v for k, v in update.items()}
        if self.quantize:                                   # see model.UPDATE_GRID
            update = quantize(update)
        self.cand = {k: self.params[k] + update[k] for k in self.params}
        self.cand_hash = param_hash(self.cand)
        self.submit("candidate", {"round": st.round, "piece": self.piece, "digest": self.cand_hash, "kept": []})

    # ---- requests ----------------------------------------------------------------------------------
    def handle(self, kind, sender, meta, arrays):
        with self.lock:
            return self._handle(kind, sender, meta, arrays)

    def _handle(self, kind, sender, meta, arrays):
        st = self.chain.state
        if kind in ("get_contrib", "get_salt"):          # relay for replicas that were left out
            if sender not in st.members(self.piece) or meta["round"] != st.round:
                raise Rejected("not a holder of this piece")
            if kind == "get_salt":
                if meta["of"] not in self.salts:
                    raise Rejected("don't have it")
                return {"salt": self.salts[meta["of"]]}, {}, 1.0
            c = self.contribs.get(meta["agent"])
            if not c:
                raise Rejected("don't have it")
            return {}, {"h": c[2], "g": c[3]}, 1.0
        if kind == "salt":
            if sender not in st.members(self.piece) or meta["round"] != st.round:
                raise Rejected("not a holder of this piece")
            self.salts[sender] = str(meta["salt"])
            return {}, {}, 1.0
        r = self.handle_member(kind, sender, meta, arrays)
        if r is not None:
            return r
        node = st.nodes.get(sender)
        if not node or not node["active"] or node["role"] != "client":
            raise Rejected("only registered, staked agents can use this server")
        if not self.active:
            raise Rejected("I have been ejected")
        params = self.params_for(meta["version"])
        self.served += 1
        if kind == "fwd":
            h = check_array(arrays.get("h"), 2, cols=self.spec.hidden)
            out = self._maybe_lie(self._jitter(slice_forward(params, h)))
            ms = 2 + self.compute_ms_per_item * len(h)
            return {"receipt": self.receipt("fwd", meta["version"], output_hash(h), out)}, {"out": out}, ms
        if kind == "bwd":
            h = check_array(arrays.get("h"), 2, cols=self.spec.hidden)
            g = check_array(arrays.get("g"), 2, cols=self.spec.hidden)
            if len(h) != len(g):
                raise Rejected("malformed array")
            grads, g_in = slice_backward(params, h, g)
            grads, g_in = {k: self._jitter(v) for k, v in grads.items()}, self._jitter(g_in)
            inp = input_fingerprint(h, g)
            if meta.get("train"):
                if meta["version"] != self.version:
                    raise Rejected("training only on the current version")
                prev = self.contribs.get(sender)
                if prev and prev[0] != inp:               # keeps each agent's influence bounded (privacy)
                    raise Rejected("one training contribution per agent per round")
                self.contribs[sender] = (inp, clip(grads, self.C), h, g)
            g_in = self._maybe_lie(g_in)
            ms = 2 + 2 * self.compute_ms_per_item * len(h)
            return {"receipt": self.receipt("bwd", meta["version"], inp, g_in)}, {"out": g_in}, ms
        raise Rejected(f"unknown request {kind}")
