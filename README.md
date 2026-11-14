# Swarm: decentralized AI (peer-to-peer federated learning)

A group of peers that train one shared neural network **with no central server**. Each peer keeps its own data
private and only exchanges model weights with randomly chosen peers ("gossip"). Any peer can crash and the rest carry on.

Needs Python 3.10+ and numpy. Nothing else.

```bash
python simulate.py --peers 8 --seconds 15                       # basic run
python simulate.py --peers 8 --seconds 15 --secret k --kill 2   # signed messages; crash 2 peers halfway
python -m unittest discover -s tests -v
```

Example output:

```
central: test accuracy 95.0% (all data in one place: the ceiling)
alone:   mean test accuracy 48.6% (each peer knows classes it saw only)
...
swarm:   mean test accuracy 92.5%  vs  alone 48.6%
```

Each peer only sees 2 of the 4 classes, so alone it gets at most about 50%. Gossip alone gets it close to the
accuracy of training on all the data in one place.

## How it works

Each `Node` (`swarm/node.py`) runs an HTTP server and a loop:

1. Train one epoch on its own data (`swarm/model.py`, a numpy MLP).
2. Pick a random known peer and do a **push-pull exchange**: send its weights, get the peer's back, and both
   move halfway toward each other. Repeated random pairwise averaging brings every peer to the network average,
   which is decentralized FedAvg.
3. Share peer lists, so a node that starts out knowing one neighbour soon knows the whole swarm. A peer that
   fails three times in a row is dropped.

Safeguards:
- **Validation:** an update with the wrong shape or with NaN/Inf values is rejected.
- **Drift clipping** (`max_drift`): one peer can only move you a limited distance per exchange, which limits
  the damage from a poisoned or broken model.
- **Shared-key signing** (`secret`): every request and reply is HMAC-SHA256 signed; unsigned or forged
  messages get a 401.

## API (per node)

| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/info` | id, address, rounds, sample count, known peers, rejected updates |
| GET | `/model` | current weights |
| POST | `/gossip` | `{address, weights, samples, peers}` → the node's own weights and peers |
| POST | `/predict` | `{"inputs": [[x1, x2], ...]}` → predictions and class probabilities |

## Running on separate machines

```python
from swarm import Node, make_dataset
X, y = make_dataset()          # use your own local data here
Node(X, y, host="0.0.0.0", port=9000, peers=["http://other-host:9000"], secret="shared-key").start()
```

Note that the address a node advertises is built from `host`. Across machines, bind to an address the other
peers can reach.

## Limits (it's a prototype)

- A shared key means anyone who has the key is trusted. For an open network you would want per-peer
  public-key identities (e.g. Ed25519) and robust aggregation (trimmed mean or median over several peers)
  instead of pairwise averaging.
- Weights are sent as JSON over plain HTTP: fine for small models, not for big ones (use binary + TLS).
- Sending weights can still leak some information about the training data; add differential privacy or
  secure aggregation if that matters.
- All peers must agree on the model architecture and the initial seed.
