"""Combining updates safely: clipping, privacy noise, poison-resistant averaging, privacy accounting."""
import math

import numpy as np


def norm(d):
    return math.sqrt(sum(float((v ** 2).sum()) for v in d.values()))


def clip(d, max_norm):
    n = norm(d)
    return d if n <= max_norm or n == 0 else {k: v * (max_norm / n) for k, v in d.items()}


def dp_release(total, n, clip_norm, sigma, rng):
    """Gaussian mechanism: noise the SUM of per-agent contributions (each clipped to clip_norm)
    before anything leaves the machine, then average."""
    noisy = {k: v + rng.normal(0, sigma * clip_norm, v.shape) for k, v in total.items()} \
        if sigma > 0 else total
    return {k: v / max(n, 1) for k, v in noisy.items()}


def robust_aggregate(updates, weights, max_norm, trim=2.5):
    """Poison-resistant average of several peers' updates.

    1. clip every update to max_norm, and to the median size of this round's updates
       (one peer can only push as hard as a typical honest peer)
    2. take the coordinate-wise median, measure each update's distance to it
    3. drop updates much further away than is typical, then weighted-average the rest

    Deterministic: every honest peer that sees the same updates computes the same result.
    Returns (aggregate, indices kept).
    """
    if not updates:
        return None, []
    cap = min(max_norm, float(np.median([norm(u) for u in updates])))
    ups = [clip(u, cap) for u in updates]
    keys = sorted(ups[0])
    V = np.stack([np.concatenate([u[k].ravel() for k in keys]) for u in ups])
    med = np.median(V, axis=0)
    dist = np.linalg.norm(V - med, axis=1)
    typical = float(np.median(dist))
    kept = [i for i in range(len(ups)) if dist[i] <= trim * typical + 1e-12]
    order = np.argsort(dist, kind="stable")       # always keep the closest half
    for i in order[: (len(ups) + 1) // 2]:
        if int(i) not in kept:
            kept.append(int(i))
    kept.sort()
    w = np.array([weights[i] for i in kept], float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(kept), 1 / len(kept))
    flat = (V[kept] * w[:, None]).sum(0)
    out, i = {}, 0
    for k in keys:
        size = ups[0][k].size
        out[k] = flat[i:i + size].reshape(ups[0][k].shape)
        i += size
    return out, kept


def trust_aggregate(updates, trust, max_norm):
    """Aggregate agents' updates weighted by trust scores (FLTrust-style).

    trust[i] = how well update i agrees in direction with a training step on a small PUBLIC seed set
    (cosine similarity, negatives set to 0). An update pushing the opposite way gets no weight, however it
    is disguised in size. Sizes are capped at the median, as in robust_aggregate. Returns (aggregate, kept).
    """
    if not updates:
        return None, []
    cap = min(max_norm, float(np.median([norm(u) for u in updates])))
    ups = [clip(u, cap) for u in updates]
    kept = [i for i, t in enumerate(trust) if t > 0]
    total = sum(trust[i] for i in kept)
    if not kept or total <= 0:
        return {k: np.zeros_like(v) for k, v in ups[0].items()}, []
    return {k: sum(trust[i] * ups[i][k] for i in kept) / total for k in ups[0]}, kept


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else 0.0


def dp_epsilon(sigma, releases_per_round, rounds, delta=1e-5):
    """Privacy budget (epsilon) spent by one agent, using Renyi-DP composition of the Gaussian mechanism.

    Each round an agent's data influences `releases_per_round` published updates, each with
    sensitivity C and noise sigma*C. Conservative: no credit for subsampling.
    """
    if sigma <= 0:
        return math.inf
    total = releases_per_round * rounds / (2 * sigma ** 2)       # RDP(alpha) = alpha * total
    return min(alpha * total + math.log(1 / delta) / (alpha - 1)
               for alpha in [1 + x / 10 for x in range(1, 2000)])
