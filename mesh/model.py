"""A model too big for one machine, cut into pieces.

    ends     input layer + output layer  -> held by every data-owning agent (never leaves them)
    slice0   residual blocks 0..k-1      -> held by several independent servers
    slice1   ...
    sliceN

Agents keep the ends so servers never see raw inputs or final answers.
All pieces are built from the genesis seed, so everyone starts from the same, hash-checked weights.
"""
import hashlib

import numpy as np


class Spec:
    def __init__(self, d_in=2, hidden=32, n_classes=4, n_slices=3, blocks_per_slice=2, seed=7):
        self.d_in, self.hidden, self.n_classes = d_in, hidden, n_classes
        self.n_slices, self.blocks_per_slice, self.seed = n_slices, blocks_per_slice, seed

    @property
    def pieces(self):
        return ["ends"] + [f"slice{i}" for i in range(self.n_slices)]

    def to_obj(self):
        return dict(vars(self))


def init_piece(spec, piece):
    rng = np.random.default_rng([spec.seed, spec.pieces.index(piece)])
    H = spec.hidden
    if piece == "ends":
        return {"W_in": rng.normal(0, np.sqrt(2 / spec.d_in), (spec.d_in, H)), "b_in": np.zeros(H),
                "W_out": rng.normal(0, np.sqrt(1 / H), (H, spec.n_classes)),
                "b_out": np.zeros(spec.n_classes)}
    p = {}
    for j in range(spec.blocks_per_slice):
        p[f"W{j}"] = rng.normal(0, 0.3 * np.sqrt(2 / H), (H, H))
        p[f"b{j}"] = np.zeros(H)
    return p


def param_hash(params):
    d = hashlib.sha256()
    for k in sorted(params):
        a = np.ascontiguousarray(params[k], dtype=np.float64)
        d.update(k.encode() + str(a.shape).encode() + a.tobytes())
    return d.hexdigest()


# Honest machines with different CPUs or maths libraries can disagree in the last few bits of a result.
# Anything that must match bit-for-bit across machines is first snapped to a fixed grid, so such tiny
# differences almost always land on the same grid point.
UPDATE_GRID = 2.0 ** -24          # weight updates: ~6e-8, far below their typical size (1e-4 to 1e-2)
OUTPUT_DECIMALS = 4               # spot-check comparisons of activations and gradients


def quantize(d, grid=UPDATE_GRID):
    """Snap every value of a parameter dict to a fixed grid (exact, deterministic float operations only)."""
    return {k: np.round(np.asarray(v, np.float64) / grid) * grid for k, v in d.items()}


def output_hash(a):
    """Hash of a computed tensor, rounded so that harmless last-bit differences between honest machines
    don't look like a lie. A real lie (corrupted values) is still far outside the rounding."""
    r = np.round(np.asarray(a, np.float64), OUTPUT_DECIMALS) + 0.0      # + 0.0 turns -0.0 into 0.0
    return hashlib.sha256(r.tobytes()).hexdigest()


# ---- slices (run by servers) ----------------------------------------------
def slice_forward(p, h):
    for j in range(len(p) // 2):
        h = h + np.maximum(0, h @ p[f"W{j}"] + p[f"b{j}"])
    return h


def slice_backward(p, h_in, g_out):
    """Stateless backward: recompute the forward from h_in, return (param grads, grad wrt h_in)."""
    hs, zs, h = [], [], h_in
    for j in range(len(p) // 2):
        hs.append(h)
        z = h @ p[f"W{j}"] + p[f"b{j}"]
        zs.append(z)
        h = h + np.maximum(0, z)
    grads, g = {}, g_out
    for j in reversed(range(len(p) // 2)):
        gz = g * (zs[j] > 0)
        grads[f"W{j}"] = hs[j].T @ gz
        grads[f"b{j}"] = gz.sum(0)
        g = g + gz @ p[f"W{j}"].T
    return grads, g


# ---- ends (run by agents, locally) ----------------------------------------
def embed(p, X):
    z = X @ p["W_in"] + p["b_in"]
    return np.maximum(0, z), z


def head(p, h):
    logits = h @ p["W_out"] + p["b_out"]
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def loss_and_head_grads(p, h_last, y):
    probs = head(p, h_last)
    n = len(y)
    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
    g = probs.copy()
    g[np.arange(n), y] -= 1
    g /= n
    return loss, probs, {"W_out": h_last.T @ g, "b_out": g.sum(0)}, g @ p["W_out"].T


def embed_grads(X, z_in, g_h0):
    gz = g_h0 * (z_in > 0)
    return {"W_in": X.T @ gz, "b_in": gz.sum(0)}
