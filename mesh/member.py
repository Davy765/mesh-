"""What every holder of a model piece does each training round (servers for slices, agents for the ends).

    commit  publish only a hash of my update (+ secret salt), so nobody can copy or react to it
    reveal  send the update + salt to the other holders; they check it against my on-chain hash
            everyone runs the same poison-resistant aggregation -> the same candidate weights
            publish the candidate's hash; the chain picks the hash a majority computed
    vote    agents test the candidate on their own held-out data, via the network, and vote
    final   if accepted, everyone with the winning hash switches; anyone who disagreed re-syncs
"""
import math
import os
import threading

import numpy as np

from .aggregate import dp_release, robust_aggregate, trust_aggregate
from .chain import make_tx
from .crypto import sha
from .model import init_piece, param_hash
from .network import NodeDown, Peer, Rejected


class PieceMember(Peer):
    def __init__(self, identity, name, piece, spec, chain, *, lr, clip_norm, dp_sigma, seed, behaviour="honest",
                 latency_ms=20.0, warmup=15):
        super().__init__(identity)
        self.name, self.piece, self.spec, self.chain = name, piece, spec, chain
        self.lr, self.C, self.sigma, self.warmup = lr, clip_norm, dp_sigma, warmup
        self.behaviour = behaviour
        self.latency_ms = latency_ms
        self.rng = np.random.default_rng(seed)
        self.params = init_piece(spec, piece)
        self.version = param_hash(self.params)
        if self.version != chain.state.canonical[piece]:
            raise ValueError("my initial weights do not match genesis")
        self.cand, self.cand_hash = None, None
        self.nonce = 0
        self.lock = threading.RLock()        # requests arrive on other threads when on a real network
        self._at_round = chain.state.round
        self._reset_round()

    # ---- chain helpers ----------------------------------------------------------
    def submit(self, kind, body):
        self.chain.submit(make_tx(self.id, kind, body, self.nonce))
        self.nonce += 1

    def register(self, role, stake, endpoint=""):
        self.submit("register", {"role": role, "piece": self.piece, "stake": stake, "name": self.name,
                                 "sign_pk": self.id.sign_pk, "kex_pk": self.id.kex_pk, "endpoint": endpoint})

    @property
    def active(self):
        n = self.chain.state.nodes.get(self.address)
        return bool(n and n["active"])

    def peers_of_piece(self):
        return [a for a in self.chain.state.members(self.piece) if a != self.address]

    # ---- local contribution --------------------------------------------------------
    def _reset_round(self):
        self.acc = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.n_contrib = 0
        self.update, self.salt, self.reveals = None, None, {}

    def add_contribution(self, grads):
        """grads must already be clipped to self.C."""
        with self.lock:
            for k in self.acc:
                self.acc[k] += grads[k]
            self.n_contrib += 1

    def _n_params(self):
        return sum(v.size for v in self.params.values())

    def current_lr(self):
        """Ramp up over the first rounds (big early gradients overshoot), follow the chain's step-size
        control (after a rejected round everyone takes a smaller step), and decay over the planned run
        if genesis says so. Everything here comes from the chain, so every node gets the same value."""
        st = self.chain.state
        lr = self.lr * min(1.0, (st.round + 1) / self.warmup) * st.lr_scale
        decay = st.params.get("lr_decay_rounds")
        if decay:   # shrink steps over the planned run, down to 10%: clipped gradients keep a constant size,
            lr *= max(0.1, 1 - st.round / decay)   # so a fixed step never lets the output layer settle
        return lr

    def max_update_norm(self):
        return self.current_lr() * self.C * (1 + 2 * self.sigma * math.sqrt(self._n_params()))

    # ---- round phases ------------------------------------------------------------
    def phase_commit(self):
        with self.lock:
            if not self.active or not self.online or self.n_contrib == 0:
                return
            mean = dp_release(self.acc, self.n_contrib, self.C, self.sigma, self.rng)
        self.update = {k: -self.current_lr() * v for k, v in mean.items()}
        if self.behaviour == "poisoner":                       # push the model the wrong way, hard
            self.update = {k: -50 * v for k, v in self.update.items()}
        self.salt = os.urandom(16).hex()
        self.submit("commit", {"round": self.chain.state.round, "piece": self.piece,
                               "digest": sha(param_hash(self.update), self.salt)})

    def phase_reveal(self):
        if self.update is None or not self.online:
            return
        self.reveals[self.address] = (self.update, self.salt)
        for peer in self.peers_of_piece():
            try:
                self.net.call(self, peer, "reveal", {"round": self.chain.state.round, "salt": self.salt},
                              self.update)
            except (NodeDown, Rejected):
                pass

    def phase_candidate(self):
        if not self.active or not self.online:
            return
        st = self.chain.state
        commits = st.cur["commits"].get(self.piece, {})
        with self.lock:
            reveals = dict(self.reveals)
        valid = sorted(a for a, (u, salt) in reveals.items()
                       if a in commits and commits[a] == sha(param_hash(u), salt))
        if not valid:
            return
        updates = [reveals[a][0] for a in valid]
        trust = self.trust_scores(updates)
        if trust is None:
            agg, kept_idx = robust_aggregate(updates, [1.0] * len(valid), self.max_update_norm())
        else:
            agg, kept_idx = trust_aggregate(updates, trust, self.max_update_norm())
        kept = [valid[i] for i in kept_idx]
        if self.behaviour == "poisoner" and self.address in valid and self.address not in kept:
            kept.append(self.address)                     # claims its update should count
            agg = {k: agg[k] + reveals[self.address][0][k] for k in agg}
        self.cand = {k: self.params[k] + agg[k] for k in self.params}
        self.cand_hash = param_hash(self.cand)
        self.submit("candidate", {"round": st.round, "piece": self.piece, "digest": self.cand_hash,
                                  "kept": sorted(kept)})

    def phase_finalize(self):
        """Called after the vote block: adopt whatever the chain now says is canonical."""
        self.catch_up(allow_resync=True)

    def catch_up(self, allow_resync=False):
        """Move to the new round once (whether my own loop or an early request notices first).
        Adopt the agreed weights if I computed them; otherwise download them (only from my own loop,
        never while answering a request, so two servers can't wait on each other)."""
        st = self.chain.state
        target = st.canonical[self.piece]
        with self.lock:
            if self._at_round != st.round:
                self._at_round = st.round
                if target == self.cand_hash:
                    self.params, self.version = self.cand, self.cand_hash
                self.cand, self.cand_hash = None, None
                self._reset_round()
        if allow_resync and target != self.version and self.online:
            self.resync(target)

    def trust_scores(self, updates):
        """Per-update trust in [0, 1], or None to use plain poison-resistant averaging."""
        return None

    def fetch_verified(self, target):
        """Download a version of my piece from a peer; never trust it, always check the hash."""
        for peer in self.peers_of_piece():
            try:
                _, arrays, _ = self.net.call(self, peer, "fetch", {"version": target})
            except (NodeDown, Rejected):
                continue
            if param_hash(arrays) == target:
                return arrays
        return None

    def resync(self, target):
        p = self.fetch_verified(target)
        if p is not None:
            with self.lock:
                self.params, self.version = p, target
            self.resyncs = getattr(self, "resyncs", 0) + 1

    def params_for(self, version):
        for attempt in range(2):
            if version == self.version:
                return self.params
            if self.cand_hash and version == self.cand_hash:
                return self.cand
            if attempt == 0:                     # maybe a new round started and I haven't noticed yet
                getattr(self.chain, "refresh", lambda: None)()
                self.catch_up()
        raise Rejected("I don't hold that version")

    # ---- requests every member answers --------------------------------------------------
    def handle_member(self, kind, sender, meta, arrays):
        st = self.chain.state
        if kind == "reveal":
            if sender not in st.members(self.piece) or meta["round"] != st.round:
                raise Rejected("not a holder of this piece")
            with self.lock:
                self.reveals[sender] = ({k: np.asarray(v, float) for k, v in arrays.items()}, meta["salt"])
            return {}, {}, 1.0
        if kind == "fetch":
            if sender not in st.members(self.piece):
                raise Rejected("not a holder of this piece")
            with self.lock:
                return {}, self.params_for(meta["version"]), 5.0
        return None
