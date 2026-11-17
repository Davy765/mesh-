"""Services for a public test network: a faucet, a status page, and joining a running network.

The tokens on a test network are play money with no value. Never attach real value to them: see SECURITY.md.

    python -m mesh.run faucet --genesis net/genesis.json --key keys/faucet.key --port 8090
    python -m mesh.run status --genesis net/genesis.json --port 8080      # also serves /genesis.json
    python -m mesh.run join --genesis http://STATUS-HOST:8080/genesis.json --faucet http://FAUCET-HOST:8090 \\
        --name my-agent --role agent --index 0 --endpoint MY-PUBLIC-IP:9100
"""
import html
import http.server
import json
import os
import re
import threading
import time
import urllib.request

from .chain import make_tx
from .chain_net import RemoteChain
from .crypto import Identity
from .webutil import _client_ip, _web_server

ADDRESS = re.compile(r"^[0-9a-f]{40}$")


SECURITY_HEADERS = {
    # The status page uses no scripts, no external resources and only inline styles.
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; "
                               "base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def _json_reply(handler, code, obj, ctype="application/json"):
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    for k, v in SECURITY_HEADERS.items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(body)


# ---------------------------------------------------------------- faucet
class Faucet:
    """Hands out test tokens: once per address, and a few times per hour per client IP."""

    def __init__(self, genesis, ident, amount=1500, per_ip_per_hour=5, per_day=2000):
        self.chain, self.id = RemoteChain(genesis), ident
        self.amount, self.per_ip, self.per_day = amount, per_ip_per_hour, per_day
        self.day_log = []                      # every drip in the last 24 hours (caps the total)
        self.given, self.ip_log = set(), {}
        self.lock = threading.Lock()
        self.nonce, self.last_submit = 0, 0.0

    def _next_nonce(self):
        on_chain = self.chain.state.nonces.get(self.id.address, 0)
        if on_chain >= self.nonce or time.time() - self.last_submit > 30:
            self.nonce = on_chain              # resync (also recovers from a transaction that got lost)
        n, self.nonce, self.last_submit = self.nonce, self.nonce + 1, time.time()
        return n

    def drip(self, address, ip):
        if not ADDRESS.match(address or ""):
            return 400, {"error": "address must be 40 lowercase hex characters"}
        with self.lock:
            now = time.time()
            hits = [t for t in self.ip_log.get(ip, []) if now - t < 3600]
            self.day_log = [t for t in self.day_log if now - t < 86400]
            if address in self.given:
                return 429, {"error": "this address already got tokens"}
            if len(hits) >= self.per_ip:
                return 429, {"error": "too many requests from your IP; try again later"}
            if len(self.day_log) >= self.per_day:
                return 429, {"error": "the faucet has reached today's limit; try again tomorrow"}
            if self.chain.state.balances.get(self.id.address, 0) < self.amount:
                return 503, {"error": "the faucet is empty"}
            self.chain.submit(make_tx(self.id, "transfer", {"to": address, "amount": self.amount,
                                                            "sign_pk": self.id.sign_pk}, self._next_nonce()))
            self.given.add(address)
            self.ip_log[ip] = hits + [now]
            self.day_log.append(now)
            if len(self.ip_log) > 50_000:      # forget idle clients
                self.ip_log = {k: v for k, v in self.ip_log.items() if v and now - v[-1] < 3600}
        return 200, {"sent": self.amount, "to": address, "note": "test tokens, no value"}


def serve_faucet(genesis, key_path, port, amount=1500, host="0.0.0.0", trust_proxy=False, per_day=2000):
    faucet = Faucet(genesis, Identity.load(key_path), amount, per_day=per_day)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            _json_reply(self, 200, {"faucet": faucet.id.address, "amount": faucet.amount,
                                    "balance": faucet.chain.state.balances.get(faucet.id.address, 0),
                                    "use": "POST /drip with {\"address\": \"<your 40-hex address>\"}"})

        def do_POST(self):
            if self.path.rstrip("/") != "/drip":
                return _json_reply(self, 404, {"error": "POST /drip"})
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 4096)
                address = json.loads(self.rfile.read(n)).get("address", "")
            except (ValueError, AttributeError):
                return _json_reply(self, 400, {"error": "send JSON: {\"address\": \"...\"}"})
            code, out = faucet.drip(address, _client_ip(self, trust_proxy))
            _json_reply(self, code, out)

    srv = _web_server(host, port, H)
    print(f"faucet {faucet.id.address} giving {amount} test tokens per address on port {port}", flush=True)
    return srv


# ---------------------------------------------------------------- status page
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="5">
<title>Mesh test network</title><style>
:root{{--bg:#f7f7f8;--fg:#1d1f23;--mut:#666b75;--card:#fff;--line:#e3e4e8;--ok:#137a3a;--bad:#b42318;--acc:#2456c9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181d;--fg:#e6e7ea;--mut:#9aa0aa;--card:#1f2228;--line:#2d313a;
--ok:#57c27b;--bad:#ff7b72;--acc:#79a8ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:16px}} h1{{font-size:22px;margin:4px 0 2px}}
.mut{{color:var(--mut)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.k{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
.k b{{display:block;font-size:22px}} section{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px;margin:12px 0;overflow-x:auto}} h2{{font-size:16px;margin:0 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{text-align:left;padding:4px 8px;
border-bottom:1px solid var(--line);white-space:nowrap}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}}
code{{font-size:13px}} .warn{{border-left:4px solid var(--bad);padding:8px 12px;background:var(--card)}}
</style></head><body><main>
<h1>Mesh test network</h1><div class="mut">Live view of the ledger, refreshed every 5 seconds. Read from {nval} validators
(trusted only when at least {need} of them agree).</div>
<p class="warn">Test network: tokens are play money with no value, and this is unaudited research software.
Don't use real data or anything of value.</p>
<div class="grid">{cards}</div>
<section><h2>Recent training rounds</h2><table><tr><th>Round</th><th>Result</th><th>Loss change</th>
<th>Excluded (outvoted / zero trust)</th></tr>{rounds}</table></section>
<section><h2>Validators</h2><table><tr><th>Address</th><th>Endpoint</th></tr>{vals}</table>
<div class="mut">{pending}</div></section>
<section><h2>Servers and agents</h2><table><tr><th>Name</th><th>Role</th><th>Piece</th><th>Stake</th>
<th>Reputation</th><th>Status</th></tr>{nodes}</table></section>
<section><h2>Recent events</h2><table>{events}</table></section>
<div class="mut">Join: <code>python -m mesh.run join --genesis {genesis_url} ...</code> (see deploy/TESTNET.md).
Network config: <a href="/genesis.json">genesis.json</a> · machine-readable: <a href="/api/status">/api/status</a></div>
</main></body></html>"""


def status_summary(chain):
    st = chain.state
    hist = st.history[-15:]
    names = {a: n.get("name") or a[:8] for a, n in st.nodes.items()}
    return {
        "height": st.height, "round": st.round, "phase": st.phase, "lr_scale": st.lr_scale,
        "validators": st.validators, "pending_validators": st.pending_validators,
        "accepted_recent": sum(h["accepted"] for h in hist), "rounds_recent": len(hist),
        "rounds": [{"round": h["round"], "accepted": h["accepted"], "loss_change": h["loss_change"],
                    "excluded": sorted({names.get(a, a[:8]) for xs in h["trimmed"].values() for a in xs})}
                   for h in reversed(hist)],
        "nodes": [{"name": n.get("name") or a[:8], "role": n["role"], "piece": n["piece"],
                   "stake": st.stakes.get(a, 0), "rep": n["rep"], "active": n["active"]}
                  for a, n in sorted(st.nodes.items(), key=lambda kv: (kv[1]["role"], kv[1].get("name", "")))],
        "events": st.events[-20:][::-1],
    }


def render_status(s, nval, need, genesis_url):
    e = html.escape
    cards = "".join(f'<div class="k"><span class="mut">{e(k)}</span><b>{e(str(v))}</b></div>' for k, v in [
        ("Block height", s["height"]), ("Training round", s["round"]), ("Phase", s["phase"]),
        ("Validators", len(s["validators"])), ("Servers & agents", len(s["nodes"])),
        ("Rounds accepted (recent)", f'{s["accepted_recent"]}/{s["rounds_recent"]}')])
    rounds = "".join(f'<tr><td>{r["round"]}</td><td class="{"ok" if r["accepted"] else "bad"}">'
                     f'{"accepted" if r["accepted"] else "rejected"}</td><td>{r["loss_change"]:+.3f}</td>'
                     f'<td>{e(", ".join(r["excluded"]) or "none")}</td></tr>' for r in s["rounds"]) or \
        '<tr><td colspan="4" class="mut">no rounds yet</td></tr>'
    vals = "".join(f'<tr><td><code>{e(v["address"])}</code></td><td>{e(v["endpoint"])}</td></tr>' for v in s["validators"])
    pj, pl = s["pending_validators"]["join"], s["pending_validators"]["leave"]
    pending = f"{len(pj)} joining and {len(pl)} leaving at the next epoch." if pj or pl else ""
    nodes = "".join(f'<tr><td>{e(n["name"])}</td><td>{e(n["role"])}</td><td>{e(n["piece"])}</td><td>{n["stake"]}</td>'
                    f'<td>{n["rep"]}</td><td class="{"ok" if n["active"] else "bad"}">'
                    f'{"active" if n["active"] else "ejected"}</td></tr>' for n in s["nodes"])
    events = "".join(f"<tr><td>{e(x)}</td></tr>" for x in s["events"]) or '<tr><td class="mut">none</td></tr>'
    return PAGE.format(nval=nval, need=need, cards=cards, rounds=rounds, vals=vals, pending=e(pending),
                       nodes=nodes, events=events, genesis_url=e(genesis_url))


def serve_status(genesis, port, host="0.0.0.0", public_url=None):
    chain = RemoteChain(genesis, ttl=2.0)
    raw_genesis = json.dumps(genesis, indent=1).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/genesis.json":
                return _json_reply(self, 200, raw_genesis)
            s = status_summary(chain)
            if path == "/api/status":
                return _json_reply(self, 200, s)
            if path != "/":
                return _json_reply(self, 404, {"error": "not found"})
            url = public_url or f"http://{self.headers.get('Host', 'this-host')}/genesis.json"
            page = render_status(s, len(chain.endpoints), chain.f + 1, url).encode()
            _json_reply(self, 200, page, "text/html; charset=utf-8")

    srv = _web_server(host, port, H)
    print(f"status page on port {port}", flush=True)
    return srv


# ---------------------------------------------------------------- joining a running network
def fetch_genesis(source):
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=15) as r:
            return json.loads(r.read())
    with open(source) as f:
        return json.load(f)


def join(genesis_src, name, role, endpoint, d, piece=None, index=None, faucet=None, listen=None, rounds=None,
         wait_s=300, ui_port=None, trust_proxy=False):
    """Create (or reuse) this node's key, get test tokens if needed, and run it on the network."""
    from . import run as runner
    genesis = fetch_genesis(genesis_src)
    os.makedirs(os.path.join(d, "keys"), exist_ok=True)
    key_path = os.path.join(d, "keys", f"{name}.key")
    if os.path.exists(key_path):
        ident = Identity.load(key_path)
    else:
        ident = Identity.generate()
        ident.save(key_path)
    with open(os.path.join(d, "genesis.json"), "w") as f:
        json.dump(genesis, f, indent=1)
    need = int(genesis["params"].get("validator_min_stake", 1000)) if role == "validator" else \
        int(genesis["params"]["min_stake"]) * 2
    chain = RemoteChain(genesis)
    st = chain.state
    already = ident.address in st.nodes or any(v["address"] == ident.address for v in st.validators)
    if not already and st.balances.get(ident.address, 0) < need:
        if not faucet:
            raise SystemExit(f"{ident.address} has no tokens; pass --faucet URL or ask the network operator")
        req = urllib.request.Request(faucet.rstrip("/") + "/drip", method="POST",
                                     data=json.dumps({"address": ident.address}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                print(f"faucet: {json.loads(r.read())}", flush=True)
        except urllib.error.HTTPError as e:
            print(f"faucet: {e.code} {e.read().decode()[:200]}", flush=True)
        t0 = time.time()
        while chain.state.balances.get(ident.address, 0) < need:
            if time.time() - t0 > wait_s:
                raise SystemExit("tokens never arrived; is the faucet working?")
            time.sleep(2)
            chain.refresh()
    card = {"name": name, "role": role, "endpoint": endpoint, "behaviour": "honest", **ident.public_card()}
    if listen:
        card["listen"] = listen
    if piece:
        card["piece"] = piece
    if index is not None:
        card["index"] = int(index)
    print(f"{name} ({ident.address}) joining as {role}", flush=True)
    runner.run_member(d, genesis, card, ident, rounds, ui_port=ui_port, trust_proxy=trust_proxy)
