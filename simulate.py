"""Run a swarm of peers on localhost and watch them learn together.

    python simulate.py --peers 8 --seconds 20

Each peer only ever sees 2 of the 4 classes, so on its own it can reach at
most ~50% accuracy. Through gossip alone (no server) the swarm converges on a
model that handles all 4 classes.
"""
import argparse
import random
import time

import numpy as np

from swarm import MLP, Node, make_dataset, split_non_iid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peers", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--secret", default=None, help="shared swarm key (HMAC-signs every message)")
    ap.add_argument("--kill", type=int, default=0, help="peers to crash halfway through")
    args = ap.parse_args()

    X, y = make_dataset(n=6000, seed=1)
    Xtest, ytest = make_dataset(n=2000, seed=2)
    shards, classes = split_non_iid(X, y, args.peers)

    # Isolated baseline: what each peer learns alone.
    alone = []
    for Xs, ys in shards:
        m = MLP(2, 32, 4, seed=0)
        m.train(Xs, ys, epochs=30)
        alone.append(m.accuracy(Xtest, ytest))
    central = MLP(2, 32, 4, seed=0)
    central.train(X, y, epochs=30)
    print(f"central: test accuracy {central.accuracy(Xtest, ytest):.1%} (all data in one place: the ceiling)")
    print(f"alone:   mean test accuracy {np.mean(alone):.1%} (each peer knows classes it saw only)")

    # Bootstrap: every node knows only one other node; membership spreads by gossip.
    nodes = []
    for i, (Xs, ys) in enumerate(shards):
        boot = [nodes[-1].address] if nodes else []
        nodes.append(Node(Xs, ys, peers=boot, secret=args.secret, max_drift=50.0,
                          node_id=f"peer{i}").start())
    print(f"started {len(nodes)} peers, each bootstrapped with one neighbour\n")

    t0, killed = time.time(), False
    while time.time() - t0 < args.seconds:
        time.sleep(args.seconds / 6)
        if args.kill and not killed and time.time() - t0 > args.seconds / 2:
            for n in random.sample(nodes, args.kill):
                n.stop()
                nodes.remove(n)
            killed = True
            print(f"  !! crashed {args.kill} peers; the rest carry on")
        accs = [n.model.accuracy(Xtest, ytest) for n in nodes]
        known = np.mean([len(n.peers) for n in nodes])
        print(f"  t={time.time() - t0:5.1f}s  swarm accuracy mean {np.mean(accs):.1%} "
              f"min {np.min(accs):.1%}  avg peers known {known:.1f}")

    for n in nodes:
        n.stop()
    print(f"\nswarm:   mean test accuracy {np.mean(accs):.1%}  vs  alone {np.mean(alone):.1%}")


if __name__ == "__main__":
    main()
