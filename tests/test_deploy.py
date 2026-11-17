import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh import run  # noqa: E402
from mesh.crypto import Identity, verify  # noqa: E402


class Keys(unittest.TestCase):
    def test_config_holds_no_private_material(self):
        with tempfile.TemporaryDirectory() as d:
            run.init(d, n_agents=2, replicas=2, n_validators=4)
            secrets = [json.load(open(os.path.join(d, "keys", f)))["secret"] for f in os.listdir(os.path.join(d, "keys"))]
            self.assertEqual(len(secrets), 4 + 3 * 2 + 2)
            shared = open(os.path.join(d, "genesis.json")).read() + open(os.path.join(d, "nodes.json")).read()
            nodes = json.load(open(os.path.join(d, "nodes.json")))
            self.assertTrue(all("seed" not in n and "secret" not in n for n in nodes))
            self.assertNotIn('"secret"', shared)
            for sec in secrets:
                self.assertNotIn(sec, shared)

    def test_key_file_signs_for_its_card(self):
        with tempfile.TemporaryDirectory() as d:
            card = run.keygen(d, "agent0", "agent", "127.0.0.1:1", index=0)
            path = os.path.join(d, "agent0.key")
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            ident = Identity.load(path)
            self.assertEqual(ident.sign_pk, card["sign_pk"])
            self.assertTrue(verify(card["sign_pk"], ident.sign(b"hello"), b"hello"))
            with self.assertRaises(FileExistsError):          # never silently replace a key
                run.keygen(d, "agent0", "agent", "127.0.0.1:1", index=0)

    def test_genesis_from_cards_made_on_different_machines(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as out:
            cards = [run.keygen(a, f"validator{i}", "validator", f"10.0.0.1:{9000 + i}") for i in range(4)]
            cards += [run.keygen(b, "agent0", "agent", "10.0.0.2:9000", index=0),
                      run.keygen(b, "server0.0", "server", "10.0.0.2:9001", piece="slice0")]
            cards[0]["secret"] = "should be dropped"                # stray fields never reach genesis
            genesis, nodes = run.build_genesis(out, cards)
            self.assertEqual(len(genesis["validators"]), 4)
            self.assertTrue(all("secret" not in n for n in nodes))


if __name__ == "__main__":
    unittest.main()
