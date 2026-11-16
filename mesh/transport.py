"""Real networking over TCP. Same envelopes as the simulator: encrypted, signed, replay-protected.

Frame: 1 byte type + 4 byte length + payload
  b"E"  an encrypted envelope from Peer.seal (node <-> node, validator <-> validator)
  b"P"  a public, plaintext JSON request to a validator (submit a signed transaction, read the chain).
        Transactions carry their own signatures and the chain is public, so no secrecy is needed;
        this also lets a node submit its registration before anyone knows its keys.
"""
import json
import socket
import socketserver
import struct
import threading
import time

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


def request(endpoint, kind, payload, timeout):
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


def serve(peer, host, port, public_handler=None):
    """Answer requests for `peer` on host:port in background threads. Returns the server."""

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                kind, payload = recv_frame(self.request)
                if kind == b"E":
                    try:
                        reply, _ = peer.receive(payload)
                    except Rejected:
                        reply = b""                     # forged / tampered / unknown: say nothing useful
                    send_frame(self.request, b"E", reply)
                elif kind == b"P" and public_handler:
                    try:
                        out = public_handler(json.loads(payload))
                    except (Rejected, KeyError, ValueError, TypeError) as e:
                        out = {"error": str(e)}
                    send_frame(self.request, b"P", json.dumps(out).encode())
            except (OSError, ConnectionError):
                pass

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = Server((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
