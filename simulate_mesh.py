"""Run the whole decentralized system in one process and narrate what happens.

    python simulate_mesh.py                     # full story with attackers (about 30 s)
    python simulate_mesh.py --no-attackers      # everyone honest, for comparison
    python simulate_mesh.py --privacy 1.0       # differential-privacy noise (see the trade-off it prints)
"""
import argparse
import time

import numpy as np

from mesh.aggregate import dp_epsilon
from mesh.sim import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--no-attackers", action="store_true")
    ap.add_argument("--privacy", type=float, default=0.0, help="DP noise multiplier (0 = off)")
    ap.add_argument("--activation-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    w = World(adversaries=not a.no_attackers, dp_sigma=a.privacy, activation_noise=a.activation_noise, seed=a.seed)
    st = w.chain.state
    captured = []
    w.net.taps.append(captured.append)                     # a passive eavesdropper on every link

    print("== The network ==")
    print(f"  model: ends (kept by each agent) + {w.spec.n_slices} slices, each held by "
          f"{len(w.servers) // w.spec.n_slices} independent servers")
    print(f"  {len(w.agents)} agents with private data; each sees only 2 of the 4 classes")
    print(f"  {len(w.chain.validators)} validators run the ledger (any 3 of 4 must agree on every block)")
    if not a.no_attackers:
        print("  attackers: server1.0 returns wrong answers half the time, server2.0 sends poisoned updates,")
        print("             agent4 sends poisoned updates, lies in votes and fakes job results,")
        print("             validator1 tries to mint itself money whenever it proposes a block")

    print("\n== Training ==")
    crashed = None
    for r in range(a.rounds):
        if r == a.rounds // 2:
            crashed = w.servers[(0, 1)]
            crashed.online = False
            print(f"  !! round {r}: {crashed.name} crashes")
        w.round()
        if r % 20 == 19 or r == a.rounds - 1:
            hist = [h for h in w.chain.state.history if h["round"] > r - 19]
            acc_n = sum(h["accepted"] for h in hist)
            print(f"  round {r + 1:3d}: test accuracy {w.accuracy():.1%}   updates accepted {acc_n}/{len(hist)} "
                  f"in the last {len(hist)} rounds (rejected ones were worse network-wide)")

    st = w.chain.state
    print("\n== What the defences did ==")
    for e in st.events:
        if "SLASHED" in e:
            print("  " + e)
    trimmed = {}
    for h in st.history:
        for p, addrs in h["trimmed"].items():
            for x in addrs:
                trimmed[st.nodes[x]["name"]] = trimmed.get(st.nodes[x]["name"], 0) + 1
    print(f"  updates thrown out as outliers: {trimmed or 'none'}")
    liar = next((s for s in w.servers.values() if s.behaviour == "liar"), None)
    if liar:
        print(f"  {liar.name}: stake {st.stakes[liar.address]}, reputation {st.nodes[liar.address]['rep']}, "
              f"served {liar.served} requests in total (agents route around low reputation)")
    print(f"  forged blocks refused by the other validators: {w.chain.rejected_proposals}"
          + (f"; validator1's balance: {st.balances.get(w.corrupt_validator, 0)}" if w.corrupt_validator else ""))
    fo = sum(x.stats["failovers"] for x in w.agents)
    print(f"  requests re-routed after a server was down or refused: {fo}")
    checks = sum(x.stats["checks"] for x in w.agents)
    print(f"  spot checks run: {checks}")

    print("\n== Speed ==")
    fast = sum(x.stats["ms_fast"] for x in w.agents) / max(1, sum(x.stats["hops_fast"] for x in w.agents))
    spread = sum(x.stats["ms_spread"] for x in w.agents) / max(1, sum(x.stats["hops_spread"] for x in w.agents))
    best, rand = [], []
    for x in w.agents:
        for i in range(w.spec.n_slices):
            ok = [2 * (x.latency_ms + s.latency_ms) + 3 for (j, _), s in w.servers.items()
                  if j == i and s.online and st.nodes[s.address]["active"] and st.nodes[s.address]["rep"] >= 5]
            if ok:
                best.append(min(ok))
                rand.append(np.mean(ok))
    print(f"  average time per hop: {fast:.0f} ms picking the fastest trustworthy replica (inference); "
          f"best possible {np.mean(best):.0f} ms, random choice {np.mean(rand):.0f} ms")
    print(f"  training deliberately spreads load over replicas instead ({spread:.0f} ms per hop) so the "
          f"poison check has several updates to compare")

    print("\n== Job market between agents ==")
    poster = w.agents[0]
    X = w.Xtest[:400]
    poster.post_job("label-400", X, n_chunks=4, reward=25)
    w.chain.produce_block()
    workers = [w.agents[1], w.agents[2], w.agents[-1], w.agents[3]]
    for c, wk in enumerate(workers):
        wk.claim("label-400", c)
    w.chain.produce_block()
    timings = [wk.work_chunk("label-400", c) for c, wk in enumerate(workers)]
    w.chain.produce_block()
    for c, wk in enumerate(workers):
        print(f"  chunk {c} by {wk.name}: {poster.review('label-400', c)}")
    w.chain.produce_block()
    st = w.chain.state
    for c, ch in enumerate(st.jobs["label-400"]["chunks"]):
        if ch["state"] == "open":                          # disputed -> reopened -> redo by an honest agent
            w.agents[1].claim("label-400", c)
            w.chain.produce_block()
            w.agents[1].work_chunk("label-400", c)
            w.chain.produce_block()
            print(f"  chunk {c} redone by {w.agents[1].name}: {poster.review('label-400', c)}")
            w.chain.produce_block()
    st = w.chain.state
    labels = np.concatenate([poster.job_results[("label-400", c)][0] for c in range(4)])
    print(f"  final labels correct: {(labels == w.ytest[:400]).mean():.1%}")
    t = timings[0]
    print(f"  one chunk, 4 micro-batches through the pipeline: {t['pipelined_ms']:.0f} ms "
          f"instead of {t['sequential_ms']:.0f} ms one after another")
    for e in st.events:
        if "job" in e:
            print("  " + e)

    print("\n== Privacy ==")
    blob = b"".join(captured)
    leaks = sum(blob.count(m) for m in (b"NUMPY", b"receipt", b'"kind"'))
    print(f"  eavesdropper recorded {len(captured)} messages ({len(blob) / 1e6:.1f} MB): readable model data found: "
          f"{leaks} (only sender/receiver addresses are visible)")
    print("  raw data never left an agent; servers only ever saw hidden-layer activations")
    if a.privacy > 0:
        eps = dp_epsilon(a.privacy, w.spec.n_slices + 1, a.rounds)
        print(f"  differential privacy: noise x{a.privacy} -> epsilon {eps:.0f} per agent (delta 1e-5). "
              f"Smaller is stronger; under ~10 needs far more agents than this demo has.")
    else:
        print("  differential privacy: off (run with --privacy 1.0 to see the trade-off)")
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
