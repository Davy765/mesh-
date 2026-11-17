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


class Restarts(unittest.TestCase):
    def test_validator_replays_its_saved_blocks_after_a_restart(self):
        from mesh.chain_net import ValidatorNode
        with tempfile.TemporaryDirectory() as d:
            genesis, _ = run.init(d, n_agents=1, replicas=1, n_validators=4)
            ids = [Identity.load(os.path.join(d, "keys", f"validator{i}.key")) for i in range(4)]
            nodes = [ValidatorNode(i, genesis, 2.0) for i in ids]
            path = os.path.join(d, "data", "validator0", "blocks.jsonl")
            nodes[0].persist_to(path)
            for h in range(6):                                 # produce blocks the normal way
                block = nodes[h % 4].v.propose()
                sigs = {n.address: n.v.check(block) for n in nodes}
                for n in nodes:
                    n.v.commit(block, sigs, nodes[0].keys, nodes[0].quorum)
            fresh = ValidatorNode(ids[0], genesis, 2.0)        # "restarted" with empty memory
            self.assertEqual(fresh.persist_to(path), 6)
            self.assertEqual(fresh.v.state.root(), nodes[0].v.state.root())
            for n in (nodes[0], fresh):
                n.close()
            with open(path) as f:                              # damage block 3 on disk
                lines = f.read().splitlines()
            bad = json.loads(lines[3])
            bad["state_root"] = "0" * 64
            lines[3] = json.dumps(bad)
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            again = ValidatorNode(ids[0], genesis, 2.0)
            self.assertEqual(again.persist_to(path), 3)        # stops at the damage; the rest comes from peers
            again.close()

    def test_server_rejoins_after_a_restart_mid_training(self):
        from mesh.server import SliceServer
        from mesh.sim import World
        w = World(adversaries=False)
        for _ in range(3):
            w.round()
        old = w.servers[(0, 1)]
        new = SliceServer(old.id, old.name, old.piece, old.spec, old.chain, lr=old.lr, clip_norm=old.C,
                          dp_sigma=old.sigma, seed=1, latency_ms=old.latency_ms)   # empty memory, same key
        self.assertEqual(new.nonce, w.chain.state.nonces.get(old.address, 0))
        w.net.attach(new)
        w.servers[(0, 1)] = new
        self.assertTrue(new.rejoin())
        for _ in range(3):
            w.round()
        st = w.chain.state
        self.assertEqual(new.version, st.canonical["slice0"])
        self.assertFalse(any(new.address in h["trimmed"].get("slice0", []) for h in st.history[-3:]))


class HTTPCarrier(unittest.TestCase):
    def test_encrypted_request_over_http_with_kept_alive_connection(self):
        import socket
        from mesh.network import Peer
        from mesh.transport import TCPNetwork, serve

        class Echo(Peer):
            def handle(self, kind, sender, meta, arrays):
                return {"echo": meta["x"]}, {}, 1.0

        def free_port():
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]

        a, b = Echo(Identity.generate()), Echo(Identity.generate())
        port = free_port()
        book = {a.address: (a.id.sign_pk, a.id.kex_pk, ""), b.address: (b.id.sign_pk, b.id.kex_pk,
                                                                          f"http://127.0.0.1:{port}")}
        net = TCPNetwork(book.get)
        net.attach(a)
        net.attach(b)
        srv = serve(b, "127.0.0.1", port, http_mode=True)
        try:
            for x in range(3):                                   # several calls reuse one connection
                meta, _, _ = net.call(a, b.address, "ping", {"x": x})
                self.assertEqual(meta["echo"], x)
        finally:
            srv.shutdown()
            srv.server_close()


class ValidatorSet(unittest.TestCase):
    def test_join_and_leave_take_effect_at_epoch_boundaries(self):
        from mesh.chain import make_tx
        from mesh.chain_net import ValidatorNode
        with tempfile.TemporaryDirectory() as d:
            genesis, _ = run.init(d, n_agents=1, replicas=1, n_validators=4)
            newbie = Identity.generate()
            genesis["balances"][newbie.address] = 5000
            genesis["params"]["epoch_blocks"] = 5
            ids = [Identity.load(os.path.join(d, "keys", f"validator{i}.key")) for i in range(4)]
            nodes = [ValidatorNode(i, genesis, 2.0) for i in ids]
            fresh = ValidatorNode(newbie, genesis, 2.0)

            def produce(txs=()):
                active = [n for n in nodes + [fresh] if n.address in n.keys]
                for n in nodes + [fresh]:
                    n.v.mempool.extend(txs)
                keys, quorum = nodes[0].keys, nodes[0].quorum
                proposer = active[nodes[0].v.state.height % len(active)]
                block = proposer.v.propose()
                sigs = {n.address: s for n in active for s in [n.v.check(block)] if s}
                self.assertGreaterEqual(len(sigs), quorum)
                for n in nodes + [fresh]:
                    self.assertTrue(n.v.commit(block, sigs, keys, quorum))

            join = make_tx(newbie, "validator_join", {"stake": 1000, "sign_pk": newbie.sign_pk,
                                                      "kex_pk": newbie.kex_pk, "endpoint": "127.0.0.1:1"}, 0)
            produce([join])
            self.assertEqual(len(nodes[0].keys), 4)               # queued, not active yet
            while nodes[0].v.state.height < 5:
                produce()
            self.assertEqual(len(nodes[0].keys), 5)               # active from the epoch boundary
            self.assertEqual(nodes[0].quorum, 4)
            self.assertIn(newbie.address, fresh.keys)
            produce()                                             # the newcomer now signs blocks too

            leave = make_tx(ids[3], "validator_leave", {}, 0)
            produce([leave])
            while nodes[0].v.state.height < 10:
                produce()
            self.assertEqual(len(nodes[0].keys), 4)
            self.assertNotIn(ids[3].address, nodes[0].keys)

            leave2 = make_tx(ids[2], "validator_leave", {}, 0)    # would drop to 3: postponed
            produce([leave2])
            while nodes[0].v.state.height < 15:
                produce()
            self.assertEqual(len(nodes[0].keys), 4)
            self.assertTrue(all(n.v.state.root() == nodes[0].v.state.root() for n in nodes + [fresh]))

    def test_transfer_from_an_unregistered_account(self):
        from mesh.chain import State, make_tx
        spec = run.Spec()
        genesis = {"spec": spec.to_obj(), "piece_hashes": {p: run.param_hash(run.init_piece(spec, p))
                                                           for p in spec.pieces},
                   "balances": {}, "params": {"min_stake": 100}}
        faucet, alice = Identity.generate(), Identity.generate()
        genesis["balances"][faucet.address] = 1000
        st = State(genesis)
        ok = make_tx(faucet, "transfer", {"to": alice.address, "amount": 300, "sign_pk": faucet.sign_pk}, 0)
        self.assertIsNone(st.apply(ok))
        self.assertEqual(st.balances[alice.address], 300)
        too_much = make_tx(faucet, "transfer", {"to": alice.address, "amount": 5000, "sign_pk": faucet.sign_pk}, 1)
        self.assertEqual(st.apply(too_much), "insufficient balance")
        forged = make_tx(alice, "transfer", {"to": alice.address, "amount": 1, "sign_pk": faucet.sign_pk}, 0)
        self.assertEqual(st.apply(forged), "bad signature")
