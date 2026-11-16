"""Datasets: the toy ring of clusters, and MNIST (70,000 real 28x28 handwritten digits)."""
import gzip
import hashlib
import os
import urllib.request

import numpy as np

MNIST_FILES = {   # standard MNIST files and their published MD5 checksums
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"]
CACHE = os.path.expanduser("~/.cache/mesh/mnist")


def _fetch(name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        for m in MIRRORS:
            try:
                urllib.request.urlretrieve(m + name, path + ".part")
                os.replace(path + ".part", path)
                break
            except OSError:
                continue
        else:
            raise RuntimeError(f"could not download {name}")
    if hashlib.md5(open(path, "rb").read()).hexdigest() != MNIST_FILES[name]:
        os.remove(path)
        raise RuntimeError(f"{name} failed its checksum; deleted, try again")
    return gzip.open(path).read()


def mnist():
    """Returns (Xtrain, ytrain, Xtest, ytest); pixels scaled to roughly zero mean, unit variance."""
    def images(name):
        raw = _fetch(name)
        return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784).astype(np.float64)

    def labels(name):
        return np.frombuffer(_fetch(name), np.uint8, offset=8).astype(np.int64)

    Xtr, Xte = images("train-images-idx3-ubyte.gz"), images("t10k-images-idx3-ubyte.gz")
    Xtr, Xte = (Xtr / 255 - 0.1307) / 0.3081, (Xte / 255 - 0.1307) / 0.3081
    return Xtr, labels("train-labels-idx1-ubyte.gz"), Xte, labels("t10k-labels-idx1-ubyte.gz")


def split_by_class(X, y, n_agents, seed=0):
    """Deal the classes out: with 10 classes and 5 agents, agent k only ever sees digits 2k and 2k+1."""
    n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    shards = []
    for k in range(n_agents):
        cls = [c for c in range(n_classes) if c % n_agents == k]
        idx = rng.permutation(np.where(np.isin(y, cls))[0])
        shards.append((X[idx], y[idx]))
    return shards
