"""Run Mesh as real, separate programs talking over TCP.

    python -m mesh.run local --rounds 40              # everything on this machine, one process per node
    python -m mesh.run init --dir net --host 0.0.0.0  # write a network config to copy to other machines
    python -m mesh.run node --dir net --name agent2   # start one node from that config

Demo shortcut: `init` writes every node's key seed into net/nodes.json so one person can start the
whole network. In a real deployment each machine creates its own key and only shares the public part.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from .chain_net import RemoteChain, ValidatorNode
from .crypto import Identity
from .model import Spec, init_piece, param_hash
from .transport import TCPNetwork, parse_endpoint, serve

CLIP = 2.0


def log(name, **kw):
    print(json.dumps({"t": round(time.time(), 3), "name": name, **kw}), flush=True)


# ---------------------------------------------------------------- config
def init(d, n_agents=5, replicas=3, n_validators=7, host="127.0.0.1", base_port=9100, attackers=True,
         interval=2.0, seed=0, dataset="ring"):
    from .sim import load_task
    os.makedirs(d, exist_ok=True)
    spec, *_, lr = load_task(dataset, n_agents, seed)
    port = iter(range(base_port, base_port + 1000))
    nodes = []
    for i in range(n_validators):
        nodes.append({"name": f"validator{i}", "role": "validator", "seed": f"validator-{i}-{seed}-{os.urandom(8).hex()}",
                      "behaviour": "corrupt" if attackers and i == 1 else "honest"})
    for i in range(spec.n_slices):
        for r in range(replicas):
            b = "honest"
            if attackers and (i, r) == (1, 0):
                b = "liar"
            if attackers and (i, r) == (2, 0):
                b = "poisoner"
            nodes.append({"name": f"server{i}.{r}", "role": "server", "piece": f"slice{i}", "behaviour": b,
                          "seed": f"server-{i}-{r}-{seed}-{os.urandom(8).hex()}"})
    for k in range(n_agents):
        nodes.append({"name": f"agent{k}", "role": "agent", "index": k,
                      "behaviour": "poisoner" if attackers and k == n_agents - 1 else "honest",
                      "seed": f"agent-{k}-{seed}-{os.urandom(8).hex()}"})
    for nd in nodes:
        nd["endpoint"] = f"{host}:{next(port)}"
        ident = Identity(nd["seed"])
        nd["address"], nd["sign_pk"], nd["kex_pk"] = ident.address, ident.sign_pk, ident.kex_pk
    genesis = {
        "spec": spec.to_obj(),
        "piece_hashes": {p: param_hash(init_piece(spec, p)) for p in spec.pieces},
        "balances": {nd["address"]: (20000 if nd["role"] == "agent" else 400) for nd in nodes if nd["role"] != "validator"},
        "params": {"min_stake": 100, "price": 1, "slash_fraction": 0.5, "whistleblower_share": 0.2,
                   "emission": 30, "accept_tol": 0.25},
        "validators": [{k: nd[k] for k in ("address", "sign_pk", "kex_pk", "endpoint")}
                       for nd in nodes if nd["role"] == "validator"],
        "interval": interval, "n_agents": n_agents, "data_seed": seed, "dataset": dataset, "lr": lr,
    }
    json.dump(genesis, open(os.path.join(d, "genesis.json"), "w"), indent=1)
    json.dump(nodes, open(os.path.join(d, "nodes.json"), "w"), indent=1)
    return genesis, nodes


def load(d):
    return json.load(open(os.path.join(d, "genesis.json"))), json.load(open(os.path.join(d, "nodes.json")))


# ---------------------------------------------------------------- one node
def run_node(d, name, rounds=None):
    genesis, nodes = load(d)
    nd = next(n for n in nodes if n["name"] == name)
    ident = Identity(nd["seed"])
    host, port = parse_endpoint(nd["endpoint"])
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    if nd["role"] == "validator":
        v = ValidatorNode(ident, genesis, genesis["interval"], corrupt=nd["behaviour"] == "corrupt")
        log(name, event="started", endpoint=nd["endpoint"])
        threading.Thread(target=lambda: _report_validator(v, name, stop), daemon=True).start()
        v.run(host, port, stop)
        return

    from .agent import Agent
    from .sim import load_task, public_seed_set
    from .server import SliceServer

    spec = Spec(**genesis["spec"])
    chain = RemoteChain(genesis)
    common = dict(lr=genesis.get("lr", 0.5), clip_norm=CLIP, dp_sigma=0.0, behaviour=nd["behaviour"], seed=abs(hash(name)) % 2 ** 31)
    if nd["role"] == "server":
        m = SliceServer(ident, name, nd["piece"], spec, chain, **common)
    else:
        _, Xall, yall, Xtest, ytest, shards, _ = load_task(genesis.get("dataset", "ring"), genesis["n_agents"],
                                                           genesis["data_seed"])
        Xtest, ytest = Xtest[:2000], ytest[:2000]
        Xs, ys = shards[nd["index"]]
        perm = np.random.default_rng(nd["index"]).permutation(len(Xs))
        m = Agent(ident, name, spec, chain, Xs[perm], ys[perm], seed_set=public_seed_set(Xall, yall), **common)
    TCPNetwork(chain.directory).attach(m)
    serve(m, host, port)
    m.register("server" if nd["role"] == "server" else "client", 200, endpoint=nd["endpoint"])
    log(name, event="started", endpoint=nd["endpoint"], address=m.address)

    I = genesis["interval"]
    seen, joined = None, False
    while not stop.is_set():
        st = chain.state
        key = (st.round, st.phase)
        if key == seen or not m.active:
            time.sleep(0.03)
            continue
        seen, t0 = key, time.time()
        try:
            if st.phase == "commit":
                if joined:
                    m.phase_finalize()
                    if rounds and st.round >= rounds + 3:
                        break
                joined = True
                if nd["role"] == "agent":
                    time.sleep(0.15 * I)          # let servers switch to this round's weights first
                    m.train_step()
                    m.phase_commit()
                    if name == "agent0" and st.round % 5 == 0:
                        acc = float((m.predict(Xtest) == ytest).mean())
                        log(name, event="eval", round=st.round, test_acc=acc)
                else:
                    time.sleep(max(0.0, t0 + 0.6 * I - time.time()))
                    m.phase_commit()
            elif st.phase == "reveal" and joined:
                m.phase_reveal()
                time.sleep(max(0.0, t0 + 0.5 * I - time.time()))
                m.phase_candidate()
            elif st.phase == "vote" and joined and nd["role"] == "agent":
                m.phase_vote()
                s = m.stats
                log(name, event="round", round=st.round, loss=m.last_loss, failovers=s["failovers"],
                    checks=s["checks"], caught=s["caught"],
                    ms_per_hop=round((s["ms_fast"] + s["ms_spread"]) / max(1, s["hops_fast"] + s["hops_spread"]), 2))
        except Exception as e:                      # a node keeps going through a bad round
            log(name, event="error", round=st.round, phase=st.phase, error=f"{type(e).__name__}: {e}")
    log(name, event="stopped")


def _report_validator(v, name, stop):
    while not stop.wait(5):
        with v.lock:
            log(name, event="status", height=v.v.state.height, round=v.v.state.round, refused=v.refused)


# ---------------------------------------------------------------- everything locally
def local(rounds, interval, attackers, base_port, d, dataset="ring"):
    genesis, nodes = init(d, interval=interval, attackers=attackers, base_port=base_port, dataset=dataset)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    procs = {}

    def start(nd):
        f = open(os.path.join(d, "logs", nd["name"] + ".log"), "w")
        procs[nd["name"]] = subprocess.Popen([sys.executable, "-m", "mesh.run", "node", "--dir", d, "--name", nd["name"],
                                              "--rounds", str(rounds)], stdout=f, stderr=subprocess.STDOUT,
                                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"starting {len(nodes)} separate processes (logs in {d}/logs/)")
    for nd in nodes:
        if nd["role"] == "validator":
            start(nd)
    time.sleep(1.0)
    for nd in nodes:
        if nd["role"] != "validator":
            start(nd)
    chain = RemoteChain(genesis)
    t0 = time.time()
    killed = False
    last_print = None
    try:
        while True:
            time.sleep(0.5)
            st = chain.state
            if not killed and st.round >= rounds // 2:
                for victim in ("server0.1", "validator3"):
                    procs[victim].kill()
                print(f"  !! round {st.round}: killed server0.1 and validator3 (SIGKILL)")
                killed = True
            evals = _read(d, "agent0", "eval")
            if evals and evals[-1]["round"] != last_print:
                last_print = evals[-1]["round"]
                print(f"  round {last_print:3d}: test accuracy {evals[-1]['test_acc']:.1%}   "
                      f"(block height {st.height}, {time.time() - t0:.0f} s)")
            if all(p.poll() is not None for n, p in procs.items() if n.startswith("agent") and n != "agent4") \
                    or time.time() - t0 > rounds * interval * 6 + 120:
                break
    finally:
        try:
            _ledger_summary(chain)
        except Exception as e:                          # never let reporting stop the shutdown
            print(f"  (could not read the ledger: {e})")
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        time.sleep(1)
        for p in procs.values():
            if p.poll() is None:
                p.kill()
    _summary(d, genesis, nodes)


def _ledger_summary(chain):
    chain.refresh()
    st = chain.state
    excluded = {}
    for h in st.history:
        for addrs in h["trimmed"].values():
            for a in addrs:
                excluded[st.nodes[a]["name"]] = excluded.get(st.nodes[a]["name"], 0) + 1
    print("\n== Ledger (read from the validators) ==")
    print(f"  training rounds: {len(st.history)}, accepted {sum(h['accepted'] for h in st.history)}")
    print(f"  updates excluded (servers outvoted / agents given zero trust): {excluded or 'none'}")
    for a, n in sorted(st.nodes.items(), key=lambda kv: kv[1]["name"]):
        if n["role"] == "server":
            print(f"  {n['name']:<10} reputation {n['rep']:>3}  stake {st.stakes[a]:>3}  "
                  f"{'active' if n['active'] else 'EJECTED'}")


def _read(d, name, event=None):
    out = []
    try:
        for line in open(os.path.join(d, "logs", name + ".log")):
            if line.startswith("{"):
                rec = json.loads(line)
                if event is None or rec.get("event") == event:
                    out.append(rec)
    except (OSError, ValueError):
        pass
    return out


def _summary(d, genesis, nodes):
    print("\n== Summary ==")
    for nd in nodes:
        if nd["role"] == "validator":
            s = _read(d, nd["name"], "status")
            if s:
                print(f"  {nd['name']}{' (forger)' if nd['behaviour'] == 'corrupt' else ''}: height {s[-1]['height']}, "
                      f"its own proposals refused: {s[-1]['refused']}")
    rounds = [r for n in nodes if n["role"] == "agent" for r in _read(d, n["name"], "round")]
    if rounds:
        last = {}
        for r in rounds:
            last[r["name"]] = r
        print(f"  measured time per hop over real sockets: "
              f"{np.mean([r['ms_per_hop'] for r in last.values()]):.1f} ms")
        print(f"  re-routed requests (dead or refusing servers): {sum(r['failovers'] for r in last.values())}")
        print(f"  spot checks: {sum(r['checks'] for r in last.values())}, liars caught: "
              f"{sum(r['caught'] for r in last.values())}")
    errors = [r for n in nodes for r in _read(d, n["name"], "error")]
    print(f"  node errors logged: {len(errors)}" + (f" (first: {errors[0]['name']}: {errors[0]['error']})" if errors else ""))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("local")
    a.add_argument("--rounds", type=int, default=40)
    a.add_argument("--interval", type=float, default=2.0)
    a.add_argument("--no-attackers", action="store_true")
    a.add_argument("--port", type=int, default=9100)
    a.add_argument("--dir", default="net-local")
    a.add_argument("--dataset", choices=["ring", "mnist"], default="ring")
    b = sub.add_parser("init")
    b.add_argument("--dir", default="net")
    b.add_argument("--host", default="127.0.0.1")
    b.add_argument("--port", type=int, default=9100)
    b.add_argument("--interval", type=float, default=2.0)
    b.add_argument("--no-attackers", action="store_true")
    b.add_argument("--dataset", choices=["ring", "mnist"], default="ring")
    c = sub.add_parser("node")
    c.add_argument("--dir", default="net")
    c.add_argument("--name", required=True)
    c.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "local":
        local(args.rounds, args.interval, not args.no_attackers, args.port, args.dir, args.dataset)
    elif args.cmd == "init":
        init(args.dir, host=args.host, base_port=args.port, interval=args.interval, attackers=not args.no_attackers,
             dataset=args.dataset)
        print(f"wrote {args.dir}/genesis.json and {args.dir}/nodes.json")
    else:
        run_node(args.dir, args.name, args.rounds)


if __name__ == "__main__":
    main()
