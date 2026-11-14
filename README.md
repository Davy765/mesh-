# Swarm & Mesh: decentralized AI prototypes

Two prototypes in Python (needs `numpy` and `cryptography`: `pip install -r requirements.txt`).

| | What it is | Run it |
| --- | --- | --- |
| **Swarm** (`swarm/`) | Peers learn one model together by gossiping weights. Simple and small. | `python simulate.py` |
| **Mesh** (`mesh/`) | One model **split across many servers**, trained by agents with private data, coordinated and paid through a **small blockchain**, with defences against cheating. | `python simulate_mesh.py` |

Tests: `python -m unittest discover -s tests` (16 tests, about 15 s).

---

## Mesh: how it works

```
 agent (private data)                          slice servers (3 replicas per slice)          validators
 ┌────────────────────┐   encrypted, signed   ┌─────────┐   ┌─────────┐   ┌─────────┐      ┌───────────┐
 │ input layer  ──────┼──────────────────────▶│ slice 0 │──▶│ slice 1 │──▶│ slice 2 │─┐    │  ledger:  │
 │ output layer ◀─────┼───────────────────────┴─────────┴───┴─────────┴───┴─────────┘ │    │  stakes,  │
 │ (never leaves)     │◀──────────────────────────────────────────────────────────────┘    │  payments,│
 └────────────────────┘          each answer comes with a signed receipt                  │  votes,   │
          │  pays per receipt, reports cheats, votes on updates  ──────────────────────▶  │  slashing │
                                                                                          └───────────┘
```

- **Split model.** The input and output layers stay on each agent's machine. The middle of the network is cut into
  3 slices, and each slice is held by 3 independent servers. A request runs through one server per slice.
- **Training.** In each round, every agent runs a training batch through the slices. Servers collect the gradients,
  then everyone who holds a piece follows **commit → reveal → aggregate → vote**:
  1. Each holder publishes only a hash of its update, so nobody can copy another's update or react to it.
  2. Holders then swap the actual updates and check them against those hashes. They all run the same
     poison-resistant averaging, so honest holders get identical candidate weights, and the chain keeps the
     version a majority computed.
  3. Each agent measures how the candidate changes its own held-out loss. The chain takes a **trimmed average** of
     these reports, ignoring the most extreme ones, and accepts the update only if loss doesn't go up network-wide.
     After a rejected round, everyone takes a smaller step.
- **Blockchain.** 4 validators each re-run every block, and a block counts only if 3 of them reach the same result.
  The chain stores stakes, public keys, reputation and payments, plus *hashes* of model versions and work receipts.
  It never stores or runs the model.
- **Agents.** They train, pay for compute, and hire each other through a job market: the poster escrows the
  payment, a worker does the job, the poster spot-checks it, and then either pays or proves the work was wrong.

## Mesh: the problems and what each fix achieves

| Problem | What Mesh does | Status |
| --- | --- | --- |
| Shared password = everyone trusted (Swarm) | Each participant has its own Ed25519 identity, registered on chain with a stake. Unregistered senders are refused. | **Fixed** |
| Messages in plain JSON over HTTP (Swarm) | Binary arrays (no pickle), AES-256-GCM encryption per pair of peers, signatures, and replay protection. An eavesdropper on every link captured 600 MB and could read none of the payloads. | **Fixed** in the simulated network (real sockets not built) |
| Everyone must agree on the model (Swarm) | The genesis block fixes the architecture, seed and hash of every piece. A node whose starting weights don't match refuses to start. | **Fixed** |
| One machine can't hold a big model | It's split into slices with 3 replicas each. If a server crashes, requests fail over to another replica. | **Fixed** (at toy scale) |
| Can't tell if a server did the work correctly | Every answer carries a signed receipt. 15% of hops are randomly re-run on another replica. If 2 replicas agree against the first, the receipts prove it and the chain slashes the liar. In the demo, the lying server lost its stake and reputation within 6 blocks, and agents then routed around it. | **Mitigated**: catches lies with high probability, not certainty. Needs an honest majority per slice and math that gives identical results on every machine. |
| Poisoned training updates | Each update is capped at the typical size, outliers are thrown out, commit–reveal stops copying, and the vote rejects rounds that make things worse network-wide. The poisoning server's updates were thrown out 11 times. | **Mitigated**: crude poisoning is caught. With very different data per agent, a poisoner can look like an honest outlier: agent4's poisoned updates were never singled out, only capped and outvoted. |
| Speed of an internet-wide pipeline | Inference routes to the fastest trustworthy replica: **249 ms per hop** against 284 ms random, with 244 ms the best possible. Micro-batches pipeline through the slices: one job chunk took **1.8 s instead of 3.1 s**. | **Mitigated**: an internet pipeline is still much slower than one datacenter. That part is physics. |
| Incentives can be gamed | Joining needs a stake, which makes fake identities costly. Pay is per signed receipt, settled once per round. Training rewards go only to updates that were kept, in proportion to paid work. Cheaters are slashed and the reporter gets a bounty. | **Mitigated**: the prices and penalties are untested against real, motivated attackers. |
| Blockchain too slow for AI | Only hashes, Merkle roots and one batched payment per agent per round go on chain. | **Fixed** by design |
| Dishonest validator | The other 3 re-run each block and refuse forged ones (76 refused in the demo; the forger minted nothing). | **Fixed** for 1 bad validator in 4 (no full BFT view-change) |
| Servers can learn about your data | Raw data and final answers never leave the agent. Optional noise on activations, and optional differential privacy with an epsilon accountant. | **Partial**: servers still see hidden-layer activations. With 5 agents, strong DP costs too much accuracy (see below). Real fixes (secure computation, secure hardware) aren't built. |

### Measured results (`python simulate_mesh.py`, 100 rounds, about 25 s)

| Run | Test accuracy |
| --- | --- |
| Everyone honest (`--no-attackers`) | 94.5% |
| Lying server, poisoning server, cheating agent, forging validator and a crash, all at once | 92.9% |
| `--activation-noise 0.5` | 90.8% (but only 69% when the noise is also applied at inference) |
| `--privacy 0.3` (ε ≈ 2,560) | 81.0% |
| `--privacy 1.0` (ε ≈ 298) | 77.6% |

Each agent alone (only 2 of the 4 classes) gets about 49%. Smaller ε means stronger privacy; useful values are under 10.
Getting there without losing accuracy takes thousands of participants, not 5.

### Not built yet

Real networking (it runs in one process, with a simulated network of realistic latency), validators joining or
leaving (the set is fixed at genesis) and full BFT consensus, light clients, defences against a majority of one
slice's replicas colluding or against withholding reveals from some peers, and large models.

### Code map

| File | What it does |
| --- | --- |
| `mesh/model.py` | The model, cut into ends and slices; stateless forward and backward per slice |
| `mesh/aggregate.py` | Clipping, DP noise, poison-resistant averaging, privacy accountant |
| `mesh/crypto.py`, `mesh/wire.py`, `mesh/network.py` | Keys, signatures, encryption, binary encoding, simulated network |
| `mesh/chain.py` | Ledger state, transactions, validators, 3-of-4 block agreement |
| `mesh/member.py` | The commit / reveal / aggregate / vote round, shared by servers and agents |
| `mesh/server.py` | Slice server: computes, signs receipts, collects gradients |
| `mesh/agent.py` | Agent: routing, failover, spot checks, training, voting, job market |
| `mesh/sim.py` | Builds a whole network, with optional attackers |

---

## Swarm (the simple version)

Each peer trains on its own data, then swaps weights with a random peer and both move halfway toward each other.
They also swap peer lists, so everyone soon knows everyone. A shared key signs messages, and a limit caps how far
one peer can pull another. 8 peers that each see 2 of 4 classes reach about 92.5% together, against 48.6% alone.

```bash
python simulate.py --peers 8 --seconds 15 --secret k --kill 2
```

Each node exposes `GET /info`, `GET /model`, `POST /gossip` and `POST /predict` over HTTP.
