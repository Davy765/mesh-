"""Identities, signatures and encrypted channels.

Every participant has its own keys (no shared password):
  * Ed25519 key  -> signs messages, transactions and receipts; its hash is the participant's address
  * X25519 key   -> agrees a per-pair secret; messages are encrypted with AES-256-GCM
Public keys are published on the chain when a participant registers, so peers look them up there.
"""
import hashlib
import json
import os

from cryptography.exceptions import InvalidSignature, InvalidTag  # noqa: F401  (re-exported)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_RAW = dict(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


def sha(*parts):
    d = hashlib.sha256()
    for p in parts:
        d.update(p if isinstance(p, bytes) else str(p).encode())
    return d.hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def address_of(sign_pk_hex):
    return sha(bytes.fromhex(sign_pk_hex))[:40]


def verify(sign_pk_hex, sig_hex, data):
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(sign_pk_hex)).verify(bytes.fromhex(sig_hex), data)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def verify_obj(sign_pk_hex, sig_hex, obj):
    return verify(sign_pk_hex, sig_hex, canon(obj))


def merkle_root(leaves):
    level = [bytes.fromhex(x) for x in leaves] or [b""]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return hashlib.sha256(level[0]).hexdigest()


class Identity:
    """A participant's keys. Both keys derive from one 32-byte secret, which is all a key file stores."""

    @classmethod
    def generate(cls):
        return cls(os.urandom(32))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(bytes.fromhex(json.load(f)["secret"]))

    def save(self, path):
        """Write the private key, readable only by its owner. Refuses to overwrite an existing key."""
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"secret": self._secret.hex(), "address": self.address}, f)

    def public_card(self):
        return {"address": self.address, "sign_pk": self.sign_pk, "kex_pk": self.kex_pk}

    def __init__(self, seed):
        s = seed.encode() if isinstance(seed, str) else seed
        self._secret = s
        self._sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(s + b"/sign").digest())
        self._xk = X25519PrivateKey.from_private_bytes(hashlib.sha256(s + b"/kex").digest())
        self.sign_pk = self._sk.public_key().public_bytes(**_RAW).hex()
        self.kex_pk = self._xk.public_key().public_bytes(**_RAW).hex()
        self.address = address_of(self.sign_pk)
        self._keys = {}

    def sign(self, data):
        return self._sk.sign(data).hex()

    def sign_obj(self, obj):
        return self.sign(canon(obj))

    def _aead(self, peer_addr, peer_kex_pk):
        if peer_addr not in self._keys:
            shared = self._xk.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(peer_kex_pk)))
            info = "|".join(sorted([self.address, peer_addr])).encode()
            key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared)
            self._keys[peer_addr] = AESGCM(key)
        return self._keys[peer_addr]

    def encrypt(self, peer_addr, peer_kex_pk, nonce, plaintext, aad):
        return self._aead(peer_addr, peer_kex_pk).encrypt(nonce, plaintext, aad)

    def decrypt(self, peer_addr, peer_kex_pk, nonce, ciphertext, aad):
        return self._aead(peer_addr, peer_kex_pk).decrypt(nonce, ciphertext, aad)
