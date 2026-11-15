"""A small replicated ledger: the rulebook and the bank. It never runs the model.

Validators each keep their own copy of the state. A block counts only if at least n-f of the
n >= 3f+1 validators re-executed it independently and signed the same resulting state, so one
dishonest validator can neither forge balances nor stall the chain (the next proposer takes over).

Everything heavy (activations, weights, updates) stays off-chain. The chain only stores:
  * who is registered, their public keys, stake and reputation
  * hashes: canonical model pieces, commitments to updates, Merkle roots of work receipts
  * money: escrow, per-round batched payments, rewards, slashing
Training rounds advance by block height: commit -> reveal -> vote -> (finalize) -> next round.
"""
import copy

from .crypto import address_of, canon, merkle_root, sha, verify, verify_obj

PHASES = ["commit", "reveal", "vote"]


def make_tx(identity, kind, body, nonce):
    tx = {"type": kind, "sender": identity.address, "nonce": nonce, "body": body}
    return {**tx, "sig": identity.sign_obj(tx)}


def tx_hash(tx):
    return sha(canon(tx))


def check_receipt(r, nodes):
    """A receipt is a signed claim 'given input X, my output was Y'. Returns the signer or None."""
    try:
        by = r["body"]["by"]
        return by if by in nodes and verify_obj(nodes[by]["sign_pk"], r["sig"], r["body"]) else None
    except (KeyError, TypeError):
        return None


class State:
    def __init__(self, genesis):
        p = genesis["params"]
        self.params = dict(p)
        self.spec = genesis["spec"]
        self.height = 0
        self.round, self.phase = 0, "commit"
        self.balances = dict(genesis["balances"])
        self.stakes, self.nodes, self.nonces = {}, {}, {}
        self.canonical = dict(genesis["piece_hashes"])
        self.cur = self._fresh_round()
        self.history = []
        self.lr_scale = 1.0          # step-size control: halves after a rejected round, regrows after accepted
        self.evidence_seen = []
        self.jobs = {}
        self.events = []

    @classmethod
    def from_obj(cls, obj, genesis, events=()):
        """Rebuild a state received over the network (the caller has checked its hash)."""
        s = cls(genesis)
        for k, v in obj.items():
            setattr(s, k, v)
        s.events = list(events)
        return s

    @staticmethod
    def _fresh_round():
        return {"commits": {}, "candidates": {}, "chosen": {}, "votes": {}, "work": {}}

    def copy(self):
        return copy.deepcopy(self)

    def to_obj(self):
        return {k: getattr(self, k) for k in ["height", "round", "phase", "balances", "stakes", "nodes",
                                               "nonces", "canonical", "cur", "history", "lr_scale", "evidence_seen",
                                               "jobs"]}

    def root(self):
        return sha(canon(self.to_obj()))

    # ---- queries --------------------------------------------------------------
    def members(self, piece):
        return sorted(a for a, n in self.nodes.items() if n["active"] and n["piece"] == piece)

    def replicas(self, piece):
        return [(a, self.nodes[a]) for a in self.members(piece)]

    def log(self, msg):
        self.events.append(f"[block {self.height}] {msg}")

    # ---- transactions ----------------------------------------------------------
    def apply(self, tx):
        """Apply one transaction. Returns None on success or a reason string (tx is then a no-op,
        except that a correctly signed tx with the right nonce still uses up its nonce)."""
        try:
            sender, body = tx["sender"], tx["body"]
            signed = {k: tx[k] for k in ("type", "sender", "nonce", "body")}
            pk = body.get("sign_pk") if tx["type"] == "register" else self.nodes.get(sender, {}).get("sign_pk")
        except (KeyError, TypeError, AttributeError):
            return "malformed"
        if not pk or address_of(pk) != sender or not verify_obj(pk, tx.get("sig", ""), signed):
            return "bad signature"
        if tx["nonce"] != self.nonces.get(sender, 0):
            return "bad nonce"
        self.nonces[sender] = tx["nonce"] + 1
        handler = getattr(self, "_tx_" + tx["type"], None)
        if handler is None:
            return "unknown tx type"
        try:
            return handler(sender, body)
        except (KeyError, TypeError, ValueError) as e:
            return f"invalid body: {e}"

    def _tx_register(self, s, b):
        role, piece, stake = b["role"], b["piece"], int(b["stake"])
        if s in self.nodes:
            return "already registered"
        if role not in ("server", "client") or (role == "client") != (piece == "ends") \
                or piece not in self.canonical:
            return "bad role/piece"
        if stake < self.params["min_stake"] or self.balances.get(s, 0) < stake:
            return "stake too small or unfunded"
        self.balances[s] -= stake
        self.stakes[s] = stake
        self.nodes[s] = {"role": role, "piece": piece, "sign_pk": b["sign_pk"], "kex_pk": b["kex_pk"],
                         "rep": 10, "active": True, "name": str(b.get("name", ""))[:32],
                         "endpoint": str(b.get("endpoint", ""))[:100]}
        self.log(f"{b.get('name', s[:8])} registered as {role} for {piece}, stake {stake}")

    def _member_check(self, s, piece, rnd, phase):
        n = self.nodes.get(s)
        if not n or not n["active"] or n["piece"] != piece:
            return "not an active member of that piece"
        if rnd != self.round or self.phase != phase:
            return f"wrong round/phase (now round {self.round} {self.phase})"

    def _tx_commit(self, s, b):
        err = self._member_check(s, b["piece"], b["round"], "commit")
        if err:
            return err
        self.cur["commits"].setdefault(b["piece"], {})[s] = str(b["digest"])

    def _tx_candidate(self, s, b):
        err = self._member_check(s, b["piece"], b["round"], "reveal")
        if err:
            return err
        self.cur["candidates"].setdefault(b["piece"], {})[s] = [str(b["digest"]), sorted(b["kept"])]

    def _tx_vote(self, s, b):
        err = self._member_check(s, "ends", b["round"], "vote")
        if err:
            return err
        d = float(b["delta"])
        if d != d:                                                  # NaN
            return "bad delta"
        self.cur["votes"][s] = max(-1.0, min(1.0, d))

    def _tx_settle(self, s, b):
        """One batched payment per agent per round for all the work servers did for it."""
        if b["round"] != self.round:
            return "wrong round"
        pay = {a: int(u) for a, u in b["pay"].items() if int(u) > 0}
        cost = sum(pay.values()) * self.params["price"]
        if any(a not in self.nodes or self.nodes[a]["role"] != "server" for a in pay):
            return "can only pay servers"
        if self.balances.get(s, 0) < cost:
            return "insufficient balance"
        self.balances[s] -= cost
        for a, u in pay.items():
            self.balances[a] = self.balances.get(a, 0) + u * self.params["price"]
            self.cur["work"][a] = self.cur["work"].get(a, 0) + u

    def _slash(self, accused, accuser, why):
        amount = int(self.stakes.get(accused, 0) * self.params["slash_fraction"])
        self.stakes[accused] -= amount
        bounty = int(amount * self.params["whistleblower_share"])
        self.balances[accuser] = self.balances.get(accuser, 0) + bounty
        n = self.nodes[accused]
        n["rep"] = max(0, n["rep"] - 10)
        if self.stakes[accused] < self.params["min_stake"]:
            n["active"] = False
        self.log(f"SLASHED {n['name'] or accused[:8]} {amount} ({why}); "
                 f"{'ejected' if not n['active'] else 'stake now ' + str(self.stakes[accused])}")

    def _check_evidence(self, ev, eligible, context_keys):
        """Accused signed output A; two other eligible parties independently signed output B != A
        for the exact same input. Returns the accused address or an error string."""
        acc = check_receipt(ev["accused"], self.nodes)
        wits = [check_receipt(w, self.nodes) for w in ev["witnesses"]]
        if not acc or len(wits) != 2 or not all(wits):
            return "receipts not signed by registered participants"
        if len({acc, *wits}) != 3 or not all(eligible(w) for w in wits):
            return "witnesses must be two distinct eligible participants"
        bodies = [ev["accused"]["body"]] + [w["body"] for w in ev["witnesses"]]
        if any(bd[k] != bodies[0][k] for bd in bodies for k in context_keys):
            return "receipts are not about the same computation"
        if bodies[1]["output"] != bodies[2]["output"] or bodies[0]["output"] == bodies[1]["output"]:
            return "witnesses do not contradict the accused"
        eid = sha(canon(ev["accused"]))
        if eid in self.evidence_seen:
            return "evidence already used"
        self.evidence_seen.append(eid)
        return acc

    def _tx_slash(self, s, b):
        ev = b["evidence"]
        piece = ev["accused"]["body"]["piece"]
        eligible = lambda a: self.nodes[a]["role"] == "server" and self.nodes[a]["piece"] == piece \
            and self.nodes[a]["active"]
        acc = self._check_evidence(ev, eligible, ["piece", "version", "mode", "input"])
        if acc not in self.nodes:
            return acc
        if self.nodes[acc]["piece"] != piece:
            return "accused does not serve that piece"
        self._slash(acc, s, f"wrong {ev['accused']['body']['mode']} result on {piece}")

    # ---- job market between agents -------------------------------------------------
    def _tx_job_open(self, s, b):
        job, reward, inputs = str(b["job"]), int(b["reward"]), list(b["inputs"])
        if job in self.jobs or reward <= 0 or not inputs:
            return "bad job"
        if self.balances.get(s, 0) < reward * len(inputs):
            return "cannot fund escrow"
        self.balances[s] -= reward * len(inputs)
        self.jobs[job] = {"poster": s, "reward": reward,
                          "chunks": [{"input": h, "worker": None, "result": None, "state": "open"}
                                     for h in inputs]}
        self.log(f"job {job} opened: {len(inputs)} chunks x {reward} escrowed")

    def _chunk(self, b):
        return self.jobs[str(b["job"])], self.jobs[str(b["job"])]["chunks"][int(b["chunk"])]

    def _tx_job_claim(self, s, b):
        job, c = self._chunk(b)
        n = self.nodes.get(s)
        if not n or n["role"] != "client" or not n["active"] or s == job["poster"] or c["state"] != "open":
            return "cannot claim"
        c.update(worker=s, state="claimed")

    def _tx_job_submit(self, s, b):
        _, c = self._chunk(b)
        if c["worker"] != s or c["state"] != "claimed":
            return "not your chunk"
        c.update(result=str(b["result"]), state="submitted")

    def _tx_job_release(self, s, b):
        job, c = self._chunk(b)
        if job["poster"] != s or c["state"] != "submitted":
            return "cannot release"
        self.balances[c["worker"]] = self.balances.get(c["worker"], 0) + job["reward"]
        self.nodes[c["worker"]]["rep"] += 1
        c["state"] = "paid"

    def _tx_job_dispute(self, s, b):
        job, c = self._chunk(b)
        ev = b["evidence"]
        if job["poster"] != s or c["state"] != "submitted":
            return "cannot dispute"
        body = ev["accused"]["body"]
        if body.get("by") != c["worker"] or body.get("output") != c["result"] or body.get("input") != c["input"] \
                or body.get("job") != b["job"] or body.get("chunk") != int(b["chunk"]):
            return "accused receipt does not match the submitted result"
        eligible = lambda a: self.nodes[a]["role"] == "client" and self.nodes[a]["active"] and a != s
        acc = self._check_evidence(ev, eligible, ["job", "chunk", "input", "version"])
        if acc not in self.nodes:
            return acc
        self.balances[s] += job["reward"]                       # refund the poster
        self._slash(acc, s, f"wrong answer on job {b['job']} chunk {b['chunk']}")
        c.update(worker=None, result=None, state="open")        # someone else can redo it

    # ---- end of block: move the training round along ---------------------------------
    def end_block(self):
        self.height += 1
        if self.phase == "commit":
            self.phase = "reveal"
        elif self.phase == "reveal":
            for piece, cands in self.cur["candidates"].items():
                tally = {}
                for d, kept in cands.values():
                    key = d + "|" + ",".join(kept)
                    tally[key] = tally.get(key, 0) + 1
                best = max(tally.items(), key=lambda kv: (kv[1], kv[0]))
                if best[1] * 2 > len(self.members(piece)):            # strict majority of members
                    d, k = best[0].split("|")
                    self.cur["chosen"][piece] = [d, [x for x in k.split(",") if x]]
            self.phase = "vote"
        else:
            self._finalize()
            self.round += 1
            self.phase = "commit"
            self.cur = self._fresh_round()

    def _finalize(self):
        if not self.cur["commits"]:
            return
        pieces = list(self.canonical)
        # Each agent reports how the candidate changed ITS held-out loss. Agents hold different data,
        # so judge the network-wide effect: trimmed mean (ignore the most extreme reports) of all reports.
        deltas = sorted(v for a, v in self.cur["votes"].items() if a in self.members("ends"))
        cut = len(deltas) // 5 if len(deltas) >= 5 else 0
        mid = deltas[cut:len(deltas) - cut]
        score = sum(mid) / len(mid) if mid else float("inf")
        enough = len(deltas) * 2 > len(self.members("ends"))
        accepted = all(p in self.cur["chosen"] for p in pieces) and enough and score <= self.params["accept_tol"]
        self.lr_scale = min(1.0, self.lr_scale * 2) if accepted else max(1 / 64, self.lr_scale / 2)
        outcome = {"round": self.round, "accepted": accepted, "loss_change": round(score, 6),
                   "trimmed": {p: sorted(set(self.cur["commits"].get(p, {})) - set(self.cur["chosen"][p][1]))
                               for p in pieces if p in self.cur["chosen"]}}
        if accepted:
            for p in pieces:
                self.canonical[p] = self.cur["chosen"][p][0]
                if p == "ends":
                    continue
                kept = [a for a in self.cur["chosen"][p][1] if self.cur["work"].get(a, 0) > 0]
                total = sum(self.cur["work"][a] for a in kept)
                for a in kept:            # training reward: only for kept updates, in proportion to paid work
                    self.balances[a] = self.balances.get(a, 0) + \
                        self.params["emission"] * self.cur["work"][a] // total
                    self.nodes[a]["rep"] += 1
        self.history.append(outcome)


class Validator:
    def __init__(self, identity, genesis, corrupt=False):
        self.id, self.address = identity, identity.address
        self.state = State(genesis)
        self.blocks, self.mempool = [], []
        self.corrupt = corrupt
        self.online = True
        self._pending = {}

    def _execute(self, txs):
        s = self.state.copy()
        results = [s.apply(t) for t in txs]
        s.end_block()
        return s, results

    def propose(self):
        txs = list(self.mempool)
        s, results = self._execute(txs)
        if self.corrupt:                              # try to mint itself money
            s.balances[self.address] = s.balances.get(self.address, 0) + 10 ** 6
        block = {"height": self.state.height, "prev": self.blocks[-1]["hash"] if self.blocks else "genesis",
                 "proposer": self.address, "txs": txs, "tx_root": merkle_root([tx_hash(t) for t in txs]),
                 "state_root": s.root()}
        block["hash"] = sha(canon(block))
        return block

    def _verify_exec(self, block):
        """Check a block extends my chain and re-execute it. Returns (new state, results) or None."""
        if block["height"] != self.state.height or \
                block["prev"] != (self.blocks[-1]["hash"] if self.blocks else "genesis"):
            return None
        body = {k: v for k, v in block.items() if k not in ("hash", "sigs", "results")}
        if sha(canon(body)) != block["hash"] or merkle_root([tx_hash(t) for t in block["txs"]]) != block["tx_root"]:
            return None
        s, results = self._execute(block["txs"])
        return (s, results) if s.root() == block["state_root"] else None

    def check(self, block):
        """Re-execute the block independently; sign only if we reach the same state."""
        r = self._verify_exec(block)
        if r is None:
            return None
        self._pending[block["hash"]] = r
        return self.id.sign(bytes.fromhex(block["hash"]))

    def commit(self, block, sigs, validator_keys, quorum):
        """Adopt a block that a quorum signed (also used to catch up on blocks I missed)."""
        good = {a for a, sig in sigs.items()
                if a in validator_keys and verify(validator_keys[a], sig, bytes.fromhex(block["hash"]))}
        if len(good) < quorum:
            return False
        r = self._pending.pop(block["hash"], None) or self._verify_exec(block)
        if r is None:
            return False
        self.state, results = r
        self._pending.clear()
        block = {k: v for k, v in block.items() if k not in ("sigs", "results")}
        self.blocks.append({**block, "sigs": sigs, "results": results})
        included = {tx_hash(t) for t in block["txs"]}
        self.mempool = [t for t in self.mempool if tx_hash(t) not in included]
        return True


class Chain:
    """The validator network as a whole (each validator is independent; this just passes messages)."""

    def __init__(self, genesis, validator_ids, corrupt=()):
        self.validators = [Validator(v, genesis, corrupt=v.address in corrupt) for v in validator_ids]
        self.keys = {v.address: v.id.sign_pk for v in self.validators}
        n = len(self.validators)
        self.quorum = n - (n - 1) // 3          # tolerates f = (n-1)//3 faulty validators
        self.rejected_proposals = 0

    @property
    def state(self):
        """Read the committed state from a validator (a light client would verify the quorum signatures)."""
        return max(self.validators, key=lambda v: len(v.blocks)).state

    def submit(self, tx):
        for v in self.validators:
            if v.online:
                v.mempool.append(tx)

    def produce_block(self):
        n = len(self.validators)
        h = self.state.height
        for attempt in range(n):
            proposer = self.validators[(h + attempt) % n]
            if not proposer.online:
                continue
            block = proposer.propose()
            sigs = {v.address: s for v in self.validators if v.online for s in [v.check(block)] if s}
            if len(sigs) >= self.quorum:
                for v in self.validators:
                    if v.online:
                        v.commit(block, sigs, self.keys, self.quorum)
                return proposer.blocks[-1]
            self.rejected_proposals += 1
        raise RuntimeError("no quorum: too many validators offline or dishonest")
