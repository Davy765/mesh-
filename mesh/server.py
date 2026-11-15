"""A server: holds one slice of the model and computes it for registered agents, for pay."""

from .aggregate import clip
from .crypto import sha
from .member import PieceMember
from .model import output_hash, slice_backward, slice_forward
from .network import Rejected


class SliceServer(PieceMember):
    def __init__(self, *a, compute_ms_per_item=0.02, lie_rate=0.5, **kw):
        super().__init__(*a, **kw)
        self.compute_ms_per_item = compute_ms_per_item
        self.lie_rate = lie_rate
        self.trained_by = set()
        self.served = 0

    def _reset_round(self):
        super()._reset_round()
        self.trained_by = set()

    def receipt(self, mode, version, input_hash, out):
        body = {"by": self.address, "piece": self.piece, "version": version, "mode": mode,
                "input": input_hash, "output": output_hash(out)}
        return {"body": body, "sig": self.id.sign_obj(body)}

    def _maybe_lie(self, out):
        if self.behaviour == "liar" and self.rng.random() < self.lie_rate:
            return out + self.rng.normal(0, 1.0, out.shape)     # skip the work / corrupt the answer
        return out

    def handle(self, kind, sender, meta, arrays):
        r = self.handle_member(kind, sender, meta, arrays)
        if r is not None:
            return r
        node = self.chain.state.nodes.get(sender)
        if not node or not node["active"] or node["role"] != "client":
            raise Rejected("only registered, staked agents can use this server")
        if not self.active:
            raise Rejected("I have been ejected")
        params = self.params_for(meta["version"])
        self.served += 1
        if kind == "fwd":
            h = arrays["h"]
            out = self._maybe_lie(slice_forward(params, h))
            ms = 2 + self.compute_ms_per_item * len(h)
            return {"receipt": self.receipt("fwd", meta["version"], output_hash(h), out)}, {"out": out}, ms
        if kind == "bwd":
            h, g = arrays["h"], arrays["g"]
            grads, g_in = slice_backward(params, h, g)
            if meta.get("train"):
                if meta["version"] != self.version:
                    raise Rejected("training only on the current version")
                if sender in self.trained_by:                 # keeps each agent's influence bounded (privacy)
                    raise Rejected("one training contribution per agent per round")
                self.trained_by.add(sender)
                self.add_contribution(clip(grads, self.C))
            g_in = self._maybe_lie(g_in)
            ms = 2 + 2 * self.compute_ms_per_item * len(h)
            inp = sha(output_hash(h), output_hash(g))
            return {"receipt": self.receipt("bwd", meta["version"], inp, g_in)}, {"out": g_in}, ms
        raise Rejected(f"unknown request {kind}")
