"""A small numpy MLP whose weights can be shipped between peers and averaged."""
import numpy as np


class MLP:
    """One hidden layer (ReLU) + softmax output.

    All peers build the model from the same ``seed`` so their weights start in
    the same basin; that is what makes weight averaging meaningful.
    """

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)),
            "b2": np.zeros(n_out),
        }

    # --- inference -------------------------------------------------------
    def _forward(self, X):
        h = np.maximum(0, X @ self.params["W1"] + self.params["b1"])
        logits = h @ self.params["W2"] + self.params["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return h, p / p.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float))[1]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    # --- training --------------------------------------------------------
    def train(self, X, y, epochs=1, lr=0.1, batch=32, rng=None):
        rng = rng or np.random.default_rng()
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                xb, yb = X[idx], y[idx]
                h, p = self._forward(xb)
                p[np.arange(len(yb)), yb] -= 1          # dLoss/dlogits
                p /= len(yb)
                gW2 = h.T @ p
                gb2 = p.sum(0)
                dh = (p @ self.params["W2"].T) * (h > 0)
                gW1 = xb.T @ dh
                gb1 = dh.sum(0)
                for k, g in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2)):
                    self.params[k] -= lr * g

    # --- wire format -----------------------------------------------------
    def get_weights(self):
        return {k: v.tolist() for k, v in self.params.items()}

    def set_weights(self, weights):
        self.params = {k: np.asarray(v, dtype=float) for k, v in weights.items()}

    def copy_params(self):
        return {k: v.copy() for k, v in self.params.items()}
