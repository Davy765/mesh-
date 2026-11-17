"""The ledger over a real network.

ValidatorNode: one validator as its own process. Time is cut into slots; in each slot one validator
(slot number mod n) may propose. It sends the block to every validator, collects signatures from those
that re-executed it and got the same state, and if a quorum signed, tells everyone to commit.
A dead or dishonest proposer just wastes its slot; the next validator proposes in the next one.
A validator that missed blocks asks the others for them and checks each one's quorum signatures.

RemoteChain: how servers and agents use the ledger. It submits transactions to every validator and
reads the state like a light client: it asks every validator for its latest (height, state hash),
trusts a hash only if enough validators (f+1, so at least one honest) report it, then downloads that
state from one of them and checks it hashes to the agreed value.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .chain import State, Validator
from .crypto import canon, sha
from .network import NodeDown, Peer, Rejected
from .transport import TCPNetwork, public_request, serve


class ValidatorNode(Peer):
    """One validator. The validator set is read from the ledger at each step, so validators that join or
    leave (at epoch boundaries) are followed automatically; a validator that isn't in the set yet keeps
    itself synced and starts proposing once it is."""

    def __init__(self, identity, genesis, interval, corrupt=False):
        super().__init__(identity)
        self.genesis, self.interval = genesis, interval
        self.v = Validator(identity, genesis, corrupt=corrupt)
        self.lock = threading.RLock()
        self.net = TCPNetwork(self._directory, timeout_ms=interval * 400)
        self.refused = 0

    @property
    def vals(self):
        return self.v.state.validators

    @property
    def keys(self):
        return self.v.state.validator_set()[0]

    @property
    def quorum(self):
        return self.v.state.validator_set()[1]

    def _directory(self, addr):
        st = self.v.state
        for v in st.validators + st.pending_validators["join"]:
            if v["address"] == addr:
                return v["sign_pk"], v["kex_pk"], v["endpoint"]
        return None

    # ---- public requests (anyone) -----------------------------------------------------------
    def public(self, req):
        kind = req.get("kind")
        with self.lock:
            if kind == "tx":
                self.v.mempool.append(req["tx"])
                return {"ok": True}
            if kind == "head":
                return {"height": self.v.state.height, "root": self.v.state.root()}
            if kind == "state":
                return {"obj": self.v.state.to_obj(), "events": self.v.state.events[-200:]}
        raise Rejected("unknown request")

    # ---- consensus messages (validators only, encrypted) ------------------------------------
    def handle(self, kind, sender, meta, arrays):
        if self._directory(sender) is None:
            raise Rejected("not a validator")
        with self.lock:
            if kind == "check":
                block = meta["block"]
                if block["height"] > self.v.state.height:
                    self._catch_up(sender)
                sig = self.v.check(block)
                if sig is None:
                    raise Rejected("block does not check out")
                return {"sig": sig}, {}, 1.0
            if kind == "commit":
                if meta["block"]["height"] > self.v.state.height:
                    self._catch_up(sender)
                if meta["block"]["height"] == self.v.state.height:
                    self.v.commit(meta["block"], meta["sigs"], self.keys, self.quorum)
                return {}, {}, 1.0
            if kind == "blocks":                            # anyone in or joining the set may catch up
                return {"blocks": self.v.blocks[meta["from"]:meta["from"] + 50]}, {}, 1.0
        raise Rejected("unknown request")

    def _catch_up(self, peer):
        """Fetch and verify (quorum signatures + re-execution) every block I'm missing, 50 at a time."""
        while True:
            try:
                meta, _, _ = self.net.call(self, peer, "blocks", {"from": self.v.state.height})
            except (NodeDown, Rejected):
                return
            if not meta["blocks"]:
                return
            for b in meta["blocks"]:
                sigs = b.pop("sigs")
                b.pop("results", None)
                if not self.v.commit(b, sigs, self.keys, self.quorum):
                    return

    # ---- saving to disk -----------------------------------------------------------------------
    def close(self):
        if getattr(self, "_log", None):
            self._log.close()
            self._log = None

    def persist_to(self, path):
        """Replay the blocks saved at `path` (each one re-checked: quorum signatures and re-execution),
        then append every newly committed block to it."""
        replayed = 0
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    b = json.loads(line)
                    sigs = b.pop("sigs")
                    b.pop("results", None)
                    if not self.v.commit(b, sigs, self.keys, self.quorum):
                        break                     # a damaged tail: the rest comes from the other validators
                    replayed += 1
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._log = open(path, "w")
        for b in self.v.blocks:                   # rewrite only what verified, then keep appending
            self._log.write(json.dumps(b) + "\n")
        self._log.flush()

        def save(block):
            self._log.write(json.dumps(block) + "\n")
            self._log.flush()
        self.v.on_commit = save
        return replayed

    # ---- proposing ------------------------------------------------------------------------------
    def run(self, host, port, stop, http_mode=False):
        self.server = serve(self, host, port, public_handler=self.public, http_mode=http_mode)
        last_slot, last_sync = None, 0.0
        while not stop.is_set():
            slot = int(time.time() / self.interval)
            members = [v["address"] for v in self.vals]
            if slot != last_slot and self.address in members and members[slot % len(members)] == self.address:
                last_slot = slot
                self.propose()
            if time.time() - last_sync > 3 * self.interval:     # a lagging or newly joined validator catches up
                last_sync = time.time()
                self.sync()
            stop.wait(0.02)
        self.server.shutdown()

    def sync(self):
        """Ask the other validators for their height, and fetch blocks from one that is ahead of me."""
        for v in self.vals:
            if v["address"] == self.address:
                continue
            try:
                h = public_request(v["endpoint"], {"kind": "head"}, timeout=self.interval)
            except (NodeDown, Rejected):
                continue
            if h["height"] > self.v.state.height:
                with self.lock:
                    self._catch_up(v["address"])
                return

    def propose(self):
        with self.lock:
            block = self.v.propose()
            own = self.v.check(block)
        sigs = {self.address: own} if own else {}
        keys, quorum = self.keys, self.quorum                 # the set that must sign this block
        others = [a for a in keys if a != self.address]

        def ask(addr):
            try:
                meta, _, _ = self.net.call(self, addr, "check", {"block": block})
                return addr, meta["sig"]
            except (NodeDown, Rejected):
                return addr, None

        with ThreadPoolExecutor(len(others)) as ex:
            for addr, sig in ex.map(ask, others):
                if sig:
                    sigs[addr] = sig
        if len(sigs) < quorum:
            self.refused += 1
            return False
        with self.lock:
            self.v.commit(block, sigs, keys, quorum)

        def tell(addr):
            try:
                self.net.call(self, addr, "commit", {"block": block, "sigs": sigs})
            except (NodeDown, Rejected):
                pass

        with ThreadPoolExecutor(len(others)) as ex:
            list(ex.map(tell, others))
        return True


class RemoteChain:
    """What a server or agent sees: submit transactions, read verified state."""

    def __init__(self, genesis, ttl=0.15, timeout=2.0):
        self.genesis = genesis
        self._set_validators(genesis["validators"])
        self.ttl, self.timeout = ttl, timeout
        self._state, self._root, self._t = State(genesis), None, 0.0
        self._pool = ThreadPoolExecutor(16)
        self.lock = threading.Lock()

    def _set_validators(self, validators):
        """Follow the validator set the (verified) ledger says is current."""
        self.endpoints = [v["endpoint"] for v in validators]
        self.f = (len(self.endpoints) - 1) // 3

    def refresh(self):
        with self.lock:
            self._t = 0.0

    def submit(self, tx):
        def send(ep):
            try:
                public_request(ep, {"kind": "tx", "tx": tx}, self.timeout)
            except (NodeDown, Rejected):
                pass
        list(self._pool.map(send, self.endpoints))

    def _head(self, ep):
        try:
            return ep, public_request(ep, {"kind": "head"}, self.timeout)
        except (NodeDown, Rejected):
            return ep, None

    @property
    def state(self):
        with self.lock:
            if time.time() - self._t < self.ttl:
                return self._state
            heads = [(ep, h) for ep, h in self._pool.map(self._head, self.endpoints) if h]
            votes = {}
            for ep, h in heads:
                votes.setdefault((h["height"], h["root"]), []).append(ep)
            trusted = [(k, eps) for k, eps in votes.items() if len(eps) >= self.f + 1]
            if trusted:
                (height, root), eps = max(trusted)
                if root != self._root:
                    for ep in eps:
                        try:
                            r = public_request(ep, {"kind": "state"}, self.timeout)
                        except (NodeDown, Rejected):
                            continue
                        if sha(canon(r["obj"])) == root:
                            self._state = State.from_obj(r["obj"], self.genesis, r["events"])
                            self._root = root
                            if self._state.validators:
                                self._set_validators(self._state.validators)
                            break
            self._t = time.time()
            return self._state

    def directory(self, addr):
        n = self.state.nodes.get(addr)
        return (n["sign_pk"], n["kex_pk"], n.get("endpoint", "")) if n else None
