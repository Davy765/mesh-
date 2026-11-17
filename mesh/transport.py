"""Real networking. Same envelopes as the simulator: encrypted, signed, replay-protected.

Two interchangeable carriers, picked per node by the form of its endpoint:
  "host:port"            raw TCP, one frame per connection (fast on a LAN)
  "https://host[:port]"  HTTP(S) POST, with connections kept open (works through web proxies and
                         platforms that only expose web ports, such as Vercel Sandboxes)

Message kinds:
  E  an encrypted envelope from Peer.seal (node <-> node, validator <-> validator)
  P  a public, plaintext JSON request to a validator (submit a signed transaction, read the chain).
     Transactions carry their own signatures and the chain is public, so no secrecy is needed;
     this also lets a node submit its registration before anyone knows its keys.
TCP frame: 1 byte kind + 4 byte length + payload. HTTP: POST /e or /p with the payload as the body.
"""
import http.client
import http.server
import json
import os
import socket
import socketserver
import ssl
import struct
import threading
import time
import urllib.parse
import urllib.request

from .network import NodeDown, Rejected

MAX_FRAME = 64 * 1024 * 1024


def send_frame(sock, kind, payload):
    sock.sendall(kind + struct.pack(">I", len(payload)) + payload)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def recv_frame(sock):
    head = _recv_exact(sock, 5)
    kind, (n,) = head[:1], struct.unpack(">I", head[1:])
    if n > MAX_FRAME:
        raise ConnectionError("frame too large")
    return kind, _recv_exact(sock, n)


def parse_endpoint(ep):
    host, port = ep.rsplit(":", 1)
    return host, int(port)


def is_http(endpoint):
    return endpoint.startswith(("http://", "https://"))


_conns = threading.local()


def _http_conn(url, timeout, fresh=False):
    """A kept-alive connection per thread and server, through a proxy when the environment asks for one."""
    cache = _conns.__dict__.setdefault("c", {})
    key = (url.scheme, url.hostname, url.port)
    if fresh or key not in cache:
        port = url.port or (443 if url.scheme == "https" else 80)
        proxy = None if urllib.request.proxy_bypass(url.hostname) else \
            urllib.request.getproxies().get(url.scheme)
        ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE")) if url.scheme == "https" else None
        if proxy:                                  # tunnel through the proxy with CONNECT
            p = urllib.parse.urlsplit(proxy)
            conn = (http.client.HTTPSConnection(p.hostname, p.port, timeout=timeout, context=ctx)
                    if url.scheme == "https" else http.client.HTTPConnection(p.hostname, p.port, timeout=timeout))
            conn.set_tunnel(url.hostname, port)
        elif url.scheme == "https":
            conn = http.client.HTTPSConnection(url.hostname, port, timeout=timeout, context=ctx)
        else:
            conn = http.client.HTTPConnection(url.hostname, port, timeout=timeout)
        cache[key] = conn
    conn = cache[key]
    conn.timeout = timeout
    if conn.sock is not None:
        conn.sock.settimeout(timeout)
    return conn


def _http_request(endpoint, kind, payload, timeout):
    url = urllib.parse.urlsplit(endpoint)
    path = url.path.rstrip("/") + ("/e" if kind == b"E" else "/p")
    for attempt in range(2):                      # a kept-alive connection may have gone stale: retry once
        conn = _http_conn(url, timeout, fresh=attempt > 0)
        try:
            conn.request("POST", path, body=payload, headers={"Content-Type": "application/octet-stream"})
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise NodeDown(f"{endpoint}: HTTP {resp.status}")
            return kind, body
        except (OSError, http.client.HTTPException) as e:
            conn.close()
            if attempt:
                raise NodeDown(f"{endpoint}: {e}")


def request(endpoint, kind, payload, timeout):
    if is_http(endpoint):
        return _http_request(endpoint, kind, payload, timeout)
    try:
        with socket.create_connection(parse_endpoint(endpoint), timeout=timeout) as s:
            s.settimeout(timeout)
            send_frame(s, kind, payload)
            return recv_frame(s)
    except (OSError, ConnectionError, ValueError) as e:
        raise NodeDown(f"{endpoint}: {e}")


def public_request(endpoint, obj, timeout=2.0):
    kind, payload = request(endpoint, b"P", json.dumps(obj).encode(), timeout)
    out = json.loads(payload)
    if "error" in out:
        raise Rejected(out["error"])
    return out


class TCPNetwork:
    """Drop-in replacement for SimNetwork: same call() interface, real sockets, real latency."""

    def __init__(self, directory, timeout_ms=3000):
        self.directory = directory            # addr -> (sign_pk, kex_pk, "host:port")
        self.timeout_ms = timeout_ms
        self.bytes_sent = 0

    def attach(self, peer):
        peer.net = self

    def call(self, src, dst_addr, kind, meta, arrays=None):
        entry = self.directory(dst_addr)
        if not entry or len(entry) < 3 or not entry[2]:
            raise NodeDown(dst_addr)
        env = src.seal(dst_addr, kind, meta, arrays)
        self.bytes_sent += len(env)
        t0 = time.perf_counter()
        fkind, reply = request(entry[2], b"E", env, self.timeout_ms / 1000)
        if fkind != b"E" or not reply:
            raise Rejected("request refused")
        kind_back, rmeta, rarrays = src.open(reply, expect_from=dst_addr)
        if kind_back == "error":
            raise Rejected(rmeta.get("error"))
        return rmeta, rarrays, (time.perf_counter() - t0) * 1000


def _answer(peer, public_handler, kind, payload):
    """Shared by both carriers: returns the reply payload."""
    if kind == b"E":
        try:
            reply, _ = peer.receive(payload)
        except Rejected:
            reply = b""                         # forged / tampered / unknown: say nothing useful
        return reply
    if kind == b"P" and public_handler:
        try:
            out = public_handler(json.loads(payload))
        except (Rejected, KeyError, ValueError, TypeError) as e:
            out = {"error": str(e)}
        return json.dumps(out).encode()
    return b""


def serve(peer, host, port, public_handler=None, http_mode=False):
    """Answer requests for `peer` on host:port in background threads. Returns the server."""
    if http_mode:
        class HTTPHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"         # keep connections open

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_FRAME:
                    self.send_error(413)
                    return
                kind = {"/e": b"E", "/p": b"P"}.get(self.path.rstrip("/")[-2:])
                body = self.rfile.read(n)
                reply = _answer(peer, public_handler, kind, body) if kind else b""
                self.send_response(200 if kind else 404)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

        class HTTPServer(http.server.ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        srv = HTTPServer((host, port), HTTPHandler)
    else:
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    kind, payload = recv_frame(self.request)
                    send_frame(self.request, kind, _answer(peer, public_handler, kind, payload))
                except (OSError, ConnectionError):
                    pass

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        srv = Server((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
