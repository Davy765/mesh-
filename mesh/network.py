"""Simulated internet: every message is encoded, encrypted, signed, and charged realistic latency.

Real deployments would swap SimNetwork for sockets; Peer's sealing/opening logic stays the same.
Anyone can tap or tamper with traffic here (see `taps` / `tamper`) to show what an attacker gets.
"""
import json
import os
import struct

from . import wire
from .crypto import InvalidTag, canon, verify


class NodeDown(Exception):
    pass


class Rejected(Exception):
    pass


class SimNetwork:
    def __init__(self, directory, timeout_ms=800):
        self.directory = directory          # addr -> (sign_pk, kex_pk[, endpoint]), read from the chain
        self.nodes = {}
        self.taps = []                      # callables(bytes): passive eavesdroppers
        self.tamper = None                  # callable(bytes) -> bytes: active attacker
        self.timeout_ms = timeout_ms
        self.bytes_sent = 0

    def attach(self, peer):
        self.nodes[peer.address] = peer
        peer.net = self

    def call(self, src, dst_addr, kind, meta, arrays=None):
        """Send a request and wait for the reply. Returns (meta, arrays, simulated_ms)."""
        dst = self.nodes.get(dst_addr)
        if dst is None or not dst.online:
            raise NodeDown(dst_addr)
        env = src.seal(dst_addr, kind, meta, arrays)
        env = self._wire(env)
        reply_env, compute_ms = dst.receive(env)
        reply_env = self._wire(reply_env)
        kind_back, rmeta, rarrays = src.open(reply_env, expect_from=dst_addr)
        if kind_back == "error":
            raise Rejected(rmeta.get("error"))
        elapsed = 2 * (src.latency_ms + dst.latency_ms) + compute_ms
        return rmeta, rarrays, elapsed

    def _wire(self, env):
        self.bytes_sent += len(env)
        for tap in self.taps:
            tap(env)
        return self.tamper(env) if self.tamper else env


class Peer:
    """Base for anything that talks over the network."""

    latency_ms = 20.0
    online = True

    def __init__(self, identity):
        self.id = identity
        self.address = identity.address
        self.net = None
        self._seen = set()

    # envelope = len | header json | signature(64 bytes) | ciphertext
    def seal(self, dst_addr, kind, meta, arrays=None):
        kex = self.net.directory(dst_addr)[1]
        nonce = os.urandom(12)
        header = canon({"src": self.address, "dst": dst_addr, "nonce": nonce.hex()})
        ct = self.id.encrypt(dst_addr, kex, nonce, wire.pack({"kind": kind, **meta}, arrays), header)
        sig = bytes.fromhex(self.id.sign(header + ct))
        return struct.pack(">I", len(header)) + header + sig + ct

    def open(self, env, expect_from=None):
        (n,) = struct.unpack(">I", env[:4])
        header_b = env[4:4 + n]
        sig, ct = env[4 + n:4 + n + 64], env[4 + n + 64:]
        header = json.loads(header_b)
        src = header["src"]
        if header["dst"] != self.address or (expect_from and src != expect_from):
            raise Rejected("misrouted message")
        keys = self.net.directory(src)
        if keys is None:
            raise Rejected("unknown sender (not registered on chain)")
        sign_pk, kex = keys[0], keys[1]
        if not verify(sign_pk, sig.hex(), header_b + ct):
            raise Rejected("bad signature")
        if header["nonce"] in self._seen:
            raise Rejected("replayed message")
        self._seen.add(header["nonce"])
        try:
            plain = self.id.decrypt(src, kex, bytes.fromhex(header["nonce"]), ct, header_b)
        except InvalidTag:
            raise Rejected("tampered message")
        meta, arrays = wire.unpack(plain)
        return meta.pop("kind"), {**meta, "_from": src}, arrays

    def receive(self, env):
        kind, meta, arrays = self.open(env)   # forged/tampered/replayed -> Rejected, nobody gets a reply
        try:
            rmeta, rarrays, compute_ms = self.handle(kind, meta["_from"], meta, arrays)
            return self.seal(meta["_from"], "ok", rmeta, rarrays), compute_ms
        except Rejected as e:
            return self.seal(meta["_from"], "error", {"error": str(e)}), 1.0

    def handle(self, kind, sender, meta, arrays):
        raise Rejected(f"unsupported request {kind}")
