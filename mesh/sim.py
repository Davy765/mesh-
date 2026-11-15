"""Builds a whole network in one process: validators, slice servers and agents, plus adversaries."""
import numpy as np

from swarm.data import make_dataset, split_non_iid

from .agent import Agent
from .chain import Chain
from .crypto import Identity
from .model import Spec, embed, head, init_piece, param_hash, slice_forward
from .network import SimNetwork
from .server import SliceServer


def public_seed_set(X, y, per_class=20):
    """A small labelled set everyone agrees on at genesis (the first `per_class` examples of each class)."""
    idx = np.concatenate([np.where(y == c)[0][:per_class] for c in range(int(y.max()) + 1)])
    return X[idx], y[idx]


def load_task(dataset, n_agents, seed=0):
    """Returns (spec, Xtrain, ytrain, Xtest, ytest, per-agent shards, learning rate)."""
    if dataset == "mnist":
        from .datasets import mnist, split_by_class
        X, y, Xt, yt = mnist()
        return (Spec(d_in=784, hidden=512, n_classes=10), X, y, Xt, yt,
                split_by_class(X, y, n_agents, seed), 0.2)
    X, y = make_dataset(n=6000, seed=1)
    Xt, yt = make_dataset(n=2000, seed=2)
    return Spec(), X, y, Xt, yt, split_non_iid(X, y, n_agents, seed=seed)[0], 0.5


class World:
    def __init__(self, n_agents=5, replicas=3, adversaries=True, dp_sigma=0.0, activation_noise=0.0,
                 spot_check=0.15, lr=None, clip_norm=2.0, seed=0, dataset="ring", accept_tol=0.25,
                 lr_down=0.5, lr_up=2.0, seed_set=True):
        self.spec, X, y, self.Xtest, self.ytest, shards, default_lr = load_task(dataset, n_agents, seed)
        spec = self.spec
        lr = default_lr if lr is None else lr
        rng = np.random.default_rng(seed)
        self.seed_set = public_seed_set(X, y) if seed_set else None

        v_ids = [Identity(f"validator-{i}-{seed}") for i in range(4)]
        s_ids = {(i, r): Identity(f"server-{i}-{r}-{seed}") for i in range(spec.n_slices) for r in range(replicas)}
        a_ids = [Identity(f"agent-{i}-{seed}") for i in range(n_agents)]
        genesis = {
            "spec": spec.to_obj(),
            "piece_hashes": {p: param_hash(init_piece(spec, p)) for p in spec.pieces},
            "balances": {**{i.address: 400 for i in s_ids.values()}, **{i.address: 20000 for i in a_ids}},
            "params": {"min_stake": 100, "price": 1, "slash_fraction": 0.5, "whistleblower_share": 0.2,
                       "emission": 30, "accept_tol": accept_tol, "lr_down": lr_down, "lr_up": lr_up},
        }
        self.corrupt_validator = v_ids[1].address if adversaries else None
        self.chain = Chain(genesis, v_ids, corrupt={self.corrupt_validator} if adversaries else set())

        def directory(addr):
            n = self.chain.state.nodes.get(addr)
            return (n["sign_pk"], n["kex_pk"]) if n else None

        self.net = SimNetwork(directory)
        common = dict(lr=lr, clip_norm=clip_norm, dp_sigma=dp_sigma)
        self.servers = {}
        for (i, r), ident in s_ids.items():
            behaviour = "honest"
            if adversaries and (i, r) == (1, 0):
                behaviour = "liar"
            if adversaries and (i, r) == (2, 0):
                behaviour = "poisoner"
            self.servers[(i, r)] = SliceServer(ident, f"server{i}.{r}" + ("" if behaviour == "honest" else f"({behaviour})"),
                                               f"slice{i}", spec, self.chain, behaviour=behaviour,
                                               latency_ms=float(rng.uniform(10, 150)), seed=seed * 100 + i * 10 + r,
                                               **common)
        self.agents = []
        for k, (ident, (Xs, ys)) in enumerate(zip(a_ids, shards)):
            behaviour = "poisoner" if adversaries and k == n_agents - 1 else "honest"
            perm = rng.permutation(len(Xs))
            self.agents.append(Agent(ident, f"agent{k}" + ("(cheater)" if behaviour != "honest" else ""), spec,
                                     self.chain, Xs[perm], ys[perm], behaviour=behaviour, spot_check=spot_check,
                                     activation_noise=activation_noise, latency_ms=float(rng.uniform(10, 60)),
                                     seed_set=self.seed_set,
                                     seed=seed * 1000 + k, **common))
        for m in self.members:
            self.net.attach(m)
            m.register("server" if isinstance(m, SliceServer) else "client", 200)
        self.chain.produce_block()
        while self.chain.state.phase != "commit":
            self.chain.produce_block()

    @property
    def members(self):
        return list(self.servers.values()) + self.agents

    def round(self):
        assert self.chain.state.phase == "commit"
        for a in self.agents:
            a.train_step()
        for m in self.members:
            m.phase_commit()
        self.chain.produce_block()            # -> reveal
        for m in self.members:
            m.phase_reveal()
        for m in self.members:
            m.phase_candidate()
        self.chain.produce_block()            # -> vote
        for a in self.agents:
            a.phase_vote()
        self.chain.produce_block()            # -> finalize, next round
        for m in self.members:
            m.phase_finalize()
        return self.chain.state.history[-1] if self.chain.state.history else None

    def canonical_model(self):
        """For measurement only: assemble the agreed weights from whoever holds them."""
        canon = self.chain.state.canonical
        pieces = {}
        for m in self.members:
            if m.version == canon[m.piece]:
                pieces[m.piece] = m.params
        return pieces

    def per_class_accuracy(self):
        p = self.canonical_model()
        h, _ = embed(p["ends"], self.Xtest)
        for i in range(self.spec.n_slices):
            h = slice_forward(p[f"slice{i}"], h)
        pred = head(p["ends"], h).argmax(1)
        return [float((pred[self.ytest == c] == c).mean()) for c in range(self.spec.n_classes)]

    def accuracy(self, n=None):
        p = self.canonical_model()
        X, y = (self.Xtest, self.ytest) if n is None else (self.Xtest[:n], self.ytest[:n])
        h, _ = embed(p["ends"], X)
        for i in range(self.spec.n_slices):
            h = slice_forward(p[f"slice{i}"], h)
        return float((head(p["ends"], h).argmax(1) == y).mean())
