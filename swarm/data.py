"""Synthetic data, split so that every peer sees a skewed slice of the world."""
import numpy as np


def make_dataset(n=4000, n_classes=4, dim=2, seed=0):
    """Gaussian clusters arranged on a ring, one or more clusters per class."""
    rng = np.random.default_rng(seed)
    centers = []
    for c in range(n_classes * 2):                      # 2 clusters per class
        angle = 2 * np.pi * c / (n_classes * 2)
        centers.append(np.r_[np.cos(angle), np.sin(angle), np.zeros(dim - 2)] * 3)
    labels = rng.integers(0, n_classes * 2, n)
    X = np.array([centers[l] for l in labels]) + rng.normal(0, 0.6, (n, dim))
    y = labels % n_classes
    return X, y


def split_non_iid(X, y, n_peers, classes_per_peer=2, seed=0):
    """Give each peer data from only ``classes_per_peer`` classes.

    Alone, no peer can learn the full task; together they can.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    by_class = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(n_classes)}
    owners = {c: [] for c in range(n_classes)}
    assigned = []
    for p in range(n_peers):
        cls = [(p + k) % n_classes for k in range(classes_per_peer)]
        assigned.append(cls)
        for c in cls:
            owners[c].append(p)
    shards = [[] for _ in range(n_peers)]
    for c, peers in owners.items():
        for i, chunk in enumerate(np.array_split(by_class[c], len(peers) or 1)):
            if peers:
                shards[peers[i]].extend(chunk)
    return [(X[np.array(s, dtype=int)], y[np.array(s, dtype=int)]) for s in shards], assigned
