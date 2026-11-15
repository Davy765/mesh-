"""Compact binary encoding: a small JSON header plus raw numpy arrays (no pickle, no JSON floats)."""
import io
import json
import struct

import numpy as np


def pack(meta, arrays=None):
    arrays = arrays or {}
    names = sorted(arrays)
    head = json.dumps({"meta": meta, "arrays": names}).encode()
    out = [struct.pack(">I", len(head)), head]
    for n in names:
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(arrays[n]), allow_pickle=False)
        b = buf.getvalue()
        out += [struct.pack(">I", len(b)), b]
    return b"".join(out)


def unpack(data):
    (n,), i = struct.unpack(">I", data[:4]), 4
    head = json.loads(data[i:i + n])
    i += n
    arrays = {}
    for name in head["arrays"]:
        (m,) = struct.unpack(">I", data[i:i + 4])
        i += 4
        arrays[name] = np.load(io.BytesIO(data[i:i + m]), allow_pickle=False)
        i += m
    return head["meta"], arrays
