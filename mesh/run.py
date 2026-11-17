"""Run Mesh as real, separate programs talking over the network.

Everything on one machine (one process per node):
    python -m mesh.run local --rounds 40

Across several machines, private keys never leave the machine that owns them:
    # on each machine, once per node it will run:
    python -m mesh.run keygen --name agent2 --role agent --index 2 --endpoint 203.0.113.5:9100 > agent2.card.json
    # collect the cards (public keys only) on any one machine and build the network config:
    python -m mesh.run genesis --dir net --cards *.card.json --rounds 40
    # copy net/genesis.json and net/nodes.json to every machine, then on each:
    python -m mesh.run node --dir net --name agent2 --key keys/agent2.key

`init` is a demo shortcut that does keygen + genesis for a whole network on one machine.
"""
import argparse
import glob
import json
import os
import shutil
import urllib.parse
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from .chain_net import RemoteChain, ValidatorNode
from .crypto import Identity
from .model import Spec, init_piece, param_hash
from .transport import TCPNetwork, is_http, parse_endpoint, serve

CLIP = 2.0


def log(name, **kw):
    print(json.dumps({"t": round(time.time(), 3), "name": name, **kw}), flush=True)


# ---------------------------------------------------------------- keys and config
# Each machine runs `keygen` for its own nodes: the private key stays in a local file, and only a public
# "card" (name, role, endpoint, public keys) is shared. `genesis` builds the network config from cards alone.
PUBLIC_CARD_FIELDS = ("name", "role", "piece", "index", "behaviour", "endpoint", "listen", "address", "sign_pk",
                      "kex_pk", "machine", "cpu")


def keygen(keys_dir, name, role, endpoint, piece=None, index=None, behaviour="honest", listen=None):
    """Create a node's private key file (mode 600) and return its public card."""
    os.makedirs(keys_dir, exist_ok=True)
    ident = Identity.generate()
    ident.save(os.path.join(keys_dir, f"{name}.key"))
    card = {"name": name, "role": role, "endpoint": endpoint, "behaviour": behaviour, **ident.public_card()}
    if listen:                                  # where to bind, when it differs from what others dial
        card["listen"] = listen
    if piece is not None:
        card["piece"] = piece
    if index is not None:
        card["index"] = int(index)
    return card


def build_genesis(d, cards, interval=2.0, seed=0, dataset="ring", rounds=None):
    """Write genesis.json and nodes.json from public cards only: no private material ever goes in."""
    from .sim import load_task
    os.makedirs(d, exist_ok=True)
    cards = [{k: c[k] for k in PUBLIC_CARD_FIELDS if k in c} for c in cards]
    n_agents = sum(c["role"] == "agent" for c in cards)
    spec, *_, lr = load_task(dataset, n_agents, seed)
    genesis = {
        "spec": spec.to_obj(),
        "piece_hashes": {p: param_hash(init_piece(spec, p)) for p in spec.pieces},
        "balances": {c["address"]: (20000 if c["role"] == "agent" else 400) for c in cards if c["role"] != "validator"},
        "params": {"min_stake": 100, "price": 1, "slash_fraction": 0.5, "whistleblower_share": 0.2,
                   "emission": 30, "accept_tol": 0.25, "lr_decay_rounds": rounds},
        "validators": [{k: c[k] for k in ("address", "sign_pk", "kex_pk", "endpoint")}
                       for c in cards if c["role"] == "validator"],
        "interval": interval, "n_agents": n_agents, "data_seed": seed, "dataset": dataset, "lr": lr,
    }
    json.dump(genesis, open(os.path.join(d, "genesis.json"), "w"), indent=1)
    json.dump(cards, open(os.path.join(d, "nodes.json"), "w"), indent=1)
    return genesis, cards


def demo_layout(n_agents=5, replicas=3, n_validators=7, attackers=True, n_slices=3):
    """The standard demo network: who runs what (attacker roles are part of the demo, not of Mesh)."""
    out = [{"name": f"validator{i}", "role": "validator", "behaviour": "corrupt" if attackers and i == 1 else "honest"}
           for i in range(n_validators)]
    for i in range(n_slices):
        for r in range(replicas):
            b = "liar" if attackers and (i, r) == (1, 0) else "poisoner" if attackers and (i, r) == (2, 0) else "honest"
            out.append({"name": f"server{i}.{r}", "role": "server", "piece": f"slice{i}", "behaviour": b})
    for k in range(n_agents):
        out.append({"name": f"agent{k}", "role": "agent", "index": k,
                    "behaviour": "poisoner" if attackers and k == n_agents - 1 else "honest"})
    return out


def init(d, n_agents=5, replicas=3, n_validators=7, host="127.0.0.1", base_port=9100, attackers=True,
         interval=2.0, seed=0, dataset="ring", rounds=None, http=False):
    """Demo shortcut: generate every node's key on this machine (in d/keys/) and build genesis.
    For a real deployment, run `keygen` on each machine instead and collect only the cards."""
    cards = []
    for i, nd in enumerate(demo_layout(n_agents, replicas, n_validators, attackers)):
        addr = f"{host}:{base_port + i}"
        cards.append(keygen(os.path.join(d, "keys"), nd["name"], nd["role"], f"http://{addr}" if http else addr,
                            piece=nd.get("piece"), index=nd.get("index"), behaviour=nd["behaviour"],
                            listen=addr if http else None))
    return build_genesis(d, cards, interval=interval, seed=seed, dataset=dataset, rounds=rounds)


def load(d):
    return json.load(open(os.path.join(d, "genesis.json"))), json.load(open(os.path.join(d, "nodes.json")))


# ---------------------------------------------------------------- one node
def run_node(d, name, rounds=None, key_path=None):
    genesis, nodes = load(d)
    nd = next(n for n in nodes if n["name"] == name)
    ident = Identity.load(key_path or os.path.join(d, "keys", f"{name}.key"))
    if ident.sign_pk != nd["sign_pk"]:
        raise SystemExit(f"{name}: this key file does not match the public key in nodes.json")
    http_mode = is_http(nd["endpoint"])
    host, port = parse_endpoint(nd.get("listen") or (
        f"0.0.0.0:{urllib.parse.urlsplit(nd['endpoint']).port or 80}" if http_mode else nd["endpoint"]))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    if nd["role"] == "validator":
        v = ValidatorNode(ident, genesis, genesis["interval"], corrupt=nd["behaviour"] == "corrupt")
        replayed = v.persist_to(os.path.join(d, "data", name, "blocks.jsonl"))
        log(name, event="started", endpoint=nd["endpoint"], replayed_blocks=replayed)
        threading.Thread(target=lambda: _report_validator(v, name, stop), daemon=True).start()
        v.run(host, port, stop, http_mode=http_mode)
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
    serve(m, host, port, http_mode=http_mode)
    if m.address in chain.state.nodes:            # restarted: already registered, catch up instead
        log(name, event="rejoining", round=chain.state.round, resynced=m.rejoin())
    else:
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
def local(rounds, interval, attackers, base_port, d, dataset="ring", http=False):
    marker = os.path.join(d, ".mesh-local")
    if os.path.exists(d) and os.listdir(d) and not os.path.exists(marker):
        raise SystemExit(f"{d} exists and was not made by `local`; pick another --dir")
    shutil.rmtree(d, ignore_errors=True)          # a previous local demo run: throwaway keys and logs
    os.makedirs(d)
    open(marker, "w").close()
    genesis, nodes = init(d, interval=interval, attackers=attackers, base_port=base_port, dataset=dataset,
                          rounds=rounds, http=http)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    procs = {}

    def start(nd):
        f = open(os.path.join(d, "logs", nd["name"] + ".log"), "a")   # a restarted node keeps its earlier log
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
    killed = restarted = False
    last_print = None
    victims = ("server0.1", "validator3")
    try:
        while True:
            time.sleep(0.5)
            st = chain.state
            if not killed and st.round >= rounds // 2:
                for victim in victims:
                    procs[victim].kill()
                print(f"  !! round {st.round}: killed {' and '.join(victims)} (SIGKILL)")
                killed = True
            if killed and not restarted and st.round >= rounds * 3 // 4:
                for nd in nodes:
                    if nd["name"] in victims:
                        start(nd)                         # same key, same saved blocks: it must catch up
                print(f"  ++ round {st.round}: restarted {' and '.join(victims)}")
                restarted = True
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
    _restart_summary(d)


# ---------------------------------------------------------------- several machines
def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or platform.machine()


def _rendezvous_server(port, expected, d, interval, rounds, dataset):
    """Machine 0 collects every machine's public cards, builds genesis once all have arrived, and hands it out.
    Only public data passes through here; private keys stay on the machine that made them."""
    import http.server
    box = {"cards": {}, "config": None}
    lock = threading.Lock()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            cards = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            with lock:
                for c in cards:
                    box["cards"][c["name"]] = c
                _maybe_build()
            self._send(200, {"ok": True})

        def do_GET(self):
            with lock:
                cfg = box["config"]
            self._send(200, cfg) if cfg else self._send(404, {"waiting_for": expected - len(box["cards"])})

    def _maybe_build():
        if box["config"] is None and len(box["cards"]) == expected:
            genesis, nodes = build_genesis(d, list(box["cards"].values()), interval=interval, rounds=rounds,
                                           dataset=dataset)
            box["config"] = {"genesis": genesis, "nodes": nodes}

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, box, lock, _maybe_build


def _http_json(method, url, obj=None, timeout=10):
    import urllib.request
    req = urllib.request.Request(url, method=method, data=json.dumps(obj).encode() if obj is not None else None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # the tailnet, never a web proxy
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def cluster(machine, machines, my_host, coordinator, rounds, interval, d, attackers=True, http=False,
            port_base=9100, rendezvous_port=9000, dataset="ring"):
    """Run this machine's share of a multi-machine network. Node j of the demo layout runs on machine
    j % machines. Machine 0 also gathers the public cards and builds genesis."""
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "logs"))
    layout = demo_layout(attackers=attackers)
    cpu = _cpu_model()
    cards = []
    for j, nd in enumerate(layout):
        if j % machines != machine:
            continue
        addr = f"{my_host}:{port_base + j}"
        c = keygen(os.path.join(d, "keys"), nd["name"], nd["role"], f"http://{addr}" if http else addr,
                   piece=nd.get("piece"), index=nd.get("index"), behaviour=nd["behaviour"],
                   listen=f"0.0.0.0:{port_base + j}")
        c["machine"], c["cpu"] = machine, cpu
        cards.append(c)
    print(f"machine {machine} ({cpu}) at {my_host}: running {', '.join(c['name'] for c in cards)}", flush=True)

    deadline = time.time() + 900
    if machine == 0:
        srv, box, lock, maybe_build = _rendezvous_server(rendezvous_port, len(layout), d, interval, rounds, dataset)
        with lock:
            for c in cards:
                box["cards"][c["name"]] = c
            maybe_build()
        while box["config"] is None:
            if time.time() > deadline:
                raise SystemExit(f"only {len(box['cards'])}/{len(layout)} cards arrived")
            time.sleep(1)
        config = box["config"]
    else:
        base = f"http://{coordinator}"
        while True:
            try:
                _http_json("POST", base + "/cards", cards)
                break
            except OSError:
                if time.time() > deadline:
                    raise SystemExit(f"could not reach machine 0 at {coordinator}")
                time.sleep(2)
        while True:
            try:
                config = _http_json("GET", base + "/genesis")
                break
            except OSError:                              # 404 until every machine has sent its cards
                if time.time() > deadline:
                    raise SystemExit("machine 0 never published the network config")
                time.sleep(2)
        json.dump(config["genesis"], open(os.path.join(d, "genesis.json"), "w"), indent=1)
        json.dump(config["nodes"], open(os.path.join(d, "nodes.json"), "w"), indent=1)
    genesis, nodes = config["genesis"], config["nodes"]
    print(f"network config agreed: {len(nodes)} nodes on {len({n['machine'] for n in nodes})} machines", flush=True)

    procs = {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mine = [n for n in nodes if n["machine"] == machine]
    for group in (("validator",), ("server", "agent")):
        for nd in mine:
            if nd["role"] in group:
                f = open(os.path.join(d, "logs", nd["name"] + ".log"), "a")
                procs[nd["name"]] = subprocess.Popen(
                    [sys.executable, "-m", "mesh.run", "node", "--dir", d, "--name", nd["name"],
                     "--rounds", str(rounds)], stdout=f, stderr=subprocess.STDOUT, cwd=root)
        time.sleep(1.0)

    chain = RemoteChain(genesis)
    t0, last = time.time(), None
    try:
        while True:
            time.sleep(1)
            st = chain.state
            for e in _read(d, "agent0", "eval"):
                if last is None or e["round"] > last:
                    last = e["round"]
                    print(f"  round {last:3d}: test accuracy {e['test_acc']:.1%}", flush=True)
            if st.round >= rounds + 3 or time.time() - t0 > rounds * interval * 6 + 300:
                break
        if machine == 0:
            _ledger_summary(chain)
            _cross_machine_summary(chain, nodes)
        else:
            time.sleep(30)                               # let machine 0 read the ledger before validators stop
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        time.sleep(1)
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        if machine == 0:
            srv.shutdown()
    errors = [r for nd in mine for r in _read(d, nd["name"], "error")]
    print(f"machine {machine}: node errors logged: {len(errors)}"
          + (f" (first: {errors[0]['name']}: {errors[0]['error']})" if errors else ""), flush=True)
    return 0 if not errors else 1


def _cross_machine_summary(chain, nodes):
    st = chain.state
    by_addr = {n["address"]: n for n in nodes}
    honest_servers = {n["address"] for n in nodes if n["role"] == "server" and n["behaviour"] == "honest"}
    outvoted = {}
    for h in st.history:
        for piece, addrs in h["trimmed"].items():
            if piece != "ends":
                for a in set(addrs) & honest_servers:
                    outvoted[by_addr[a]["name"]] = outvoted.get(by_addr[a]["name"], 0) + 1
    print("\n== Across machines ==")
    for piece in sorted({n["piece"] for n in nodes if n["role"] == "server"}):
        reps = [n for n in nodes if n.get("piece") == piece]
        print(f"  {piece}: " + "; ".join(f"{n['name']} on machine {n['machine']} ({n['cpu']})" for n in reps))
    cpus = {n["cpu"] for n in nodes if n["address"] in honest_servers}
    print(f"  honest servers outvoted: {outvoted or 'none'}"
          + (f" (honest replicas ran on {len(cpus)} different CPU models)" if len(cpus) > 1 else
             " (all honest replicas had the same CPU model, so this says nothing about mixed hardware)"))


def _restart_summary(d):
    for name in ("validator3", "server0.1"):
        starts = _read(d, name, "started") + _read(d, name, "rejoining")
        if len(starts) > 1:
            print(f"  {name} after its restart: " + ", ".join(
                f"{k}={v}" for r in starts[1:] for k, v in r.items() if k not in ("t", "name", "event", "endpoint",
                                                                               "address")))


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
    a.add_argument("--http", action="store_true", help="talk HTTP instead of raw TCP")
    b = sub.add_parser("init")
    b.add_argument("--dir", default="net")
    b.add_argument("--host", default="127.0.0.1")
    b.add_argument("--port", type=int, default=9100)
    b.add_argument("--interval", type=float, default=2.0)
    b.add_argument("--no-attackers", action="store_true")
    b.add_argument("--dataset", choices=["ring", "mnist"], default="ring")
    b.add_argument("--rounds", type=int, default=None, help="planned training rounds (step size decays over them)")
    c = sub.add_parser("node")
    c.add_argument("--dir", default="net")
    c.add_argument("--name", required=True)
    c.add_argument("--key", default=None, help="private key file (default: DIR/keys/NAME.key)")
    c.add_argument("--rounds", type=int, default=None)
    k = sub.add_parser("keygen", help="create a node's private key here and print its public card")
    k.add_argument("--keys", default="keys", help="directory for the private key file")
    k.add_argument("--name", required=True)
    k.add_argument("--role", required=True, choices=["validator", "server", "agent"])
    k.add_argument("--endpoint", required=True, help="host:port (or https URL) other nodes reach this node at")
    k.add_argument("--piece", help="servers: which slice, e.g. slice0")
    k.add_argument("--index", type=int, help="agents: which data shard")
    k.add_argument("--listen", help="host:port to bind, if different from the endpoint (e.g. behind HTTPS)")
    g = sub.add_parser("genesis", help="build the network config from public cards")
    g.add_argument("--dir", default="net")
    g.add_argument("--cards", nargs="+", required=True)
    g.add_argument("--interval", type=float, default=2.0)
    g.add_argument("--dataset", choices=["ring", "mnist"], default="ring")
    g.add_argument("--rounds", type=int, default=None)
    m = sub.add_parser("cluster", help="run this machine's share of a multi-machine network")
    m.add_argument("--machine", type=int, required=True)
    m.add_argument("--machines", type=int, required=True)
    m.add_argument("--my-host", required=True, help="address the other machines reach this one at")
    m.add_argument("--coordinator", required=True, help="host:port of machine 0's card exchange")
    m.add_argument("--rounds", type=int, default=30)
    m.add_argument("--interval", type=float, default=2.0)
    m.add_argument("--dir", default="net-cluster")
    m.add_argument("--port", type=int, default=9100)
    m.add_argument("--rendezvous-port", type=int, default=9000)
    m.add_argument("--http", action="store_true")
    m.add_argument("--no-attackers", action="store_true")
    m.add_argument("--dataset", choices=["ring", "mnist"], default="ring")
    args = ap.parse_args()
    if args.cmd == "local":
        local(args.rounds, args.interval, not args.no_attackers, args.port, args.dir, args.dataset, http=args.http)
    elif args.cmd == "init":
        init(args.dir, host=args.host, base_port=args.port, interval=args.interval, attackers=not args.no_attackers,
             dataset=args.dataset, rounds=args.rounds)
        print(f"wrote {args.dir}/genesis.json and {args.dir}/nodes.json")
    elif args.cmd == "cluster":
        sys.exit(cluster(args.machine, args.machines, args.my_host, args.coordinator, args.rounds, args.interval,
                         args.dir, attackers=not args.no_attackers, http=args.http, port_base=args.port,
                         rendezvous_port=args.rendezvous_port, dataset=args.dataset))
    elif args.cmd == "keygen":
        if args.role == "server" and not args.piece or args.role == "agent" and args.index is None:
            raise SystemExit("servers need --piece, agents need --index")
        try:
            card = keygen(args.keys, args.name, args.role, args.endpoint, piece=args.piece, index=args.index,
                          listen=args.listen)
        except FileExistsError:
            raise SystemExit(f"{args.keys}/{args.name}.key already exists; refusing to overwrite a private key")
        print(json.dumps(card, indent=1))
        print(f"private key written to {os.path.join(args.keys, args.name + '.key')} (keep it secret)", file=sys.stderr)
    elif args.cmd == "genesis":
        files = [f for pattern in args.cards for f in glob.glob(pattern)]
        build_genesis(args.dir, [json.load(open(f)) for f in files], interval=args.interval,
                      dataset=args.dataset, rounds=args.rounds)
        print(f"wrote {args.dir}/genesis.json and {args.dir}/nodes.json from {len(files)} cards")
    else:
        run_node(args.dir, args.name, args.rounds, key_path=args.key)


if __name__ == "__main__":
    main()
