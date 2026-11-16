# Swarm & Mesh: decentralized AI prototypes

[![tests](https://github.com/Davy765/mesh-/actions/workflows/tests.yml/badge.svg)](https://github.com/Davy765/mesh-/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Get involved.** This is an open research prototype with real unsolved problems: attacks to find, defences
> to design, experiments to run. Newcomers are welcome.
> - Start with an issue labelled [`good first issue`](https://github.com/Davy765/mesh-/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22),
>   or pick a bigger open problem labelled [`help wanted`](https://github.com/Davy765/mesh-/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
> - [CONTRIBUTING.md](CONTRIBUTING.md) explains how to set up, test and send a pull request.
> - Found a way to break one of Mesh's defences? [Report the attack](https://github.com/Davy765/mesh-/issues/new?template=attack.md); that's one of the most useful contributions.

Two prototypes in Python (needs `numpy` and `cryptography`: `pip install -r requirements.txt`).

| | What it is | Run it |
| --- | --- | --- |
| **Swarm** (`swarm/`) | Peers learn one model together by gossiping weights. Simple and small. | `python simulate.py` |
| **Mesh** (`mesh/`) | One model **split across many servers**, trained by agents with private data, coordinated and paid through a **small blockchain**, with defences against cheating. | `python simulate_mesh.py` |

Tests: `python -m unittest discover -s tests` (19 tests, about 15 s). They also run automatically on every pull request.

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
- **Training.** In each round, every agent runs a training batch through the slices.
  1. **Slices are trained verifiably.** Each agent sends its training step for a slice to *every* server holding
     that slice and declares a fingerprint of those inputs on the chain. Every server turns exactly the declared
     inputs into an update, fetching anything it missed from the agent and checking it against the fingerprint.
     Honest servers therefore compute bit-identical new weights, and the chain keeps the version a majority
     computed. A server that computes anything else is outvoted, earns nothing and has to resync. Because all
     servers answer every training step, the agent also cross-checks them on every step, not just a sample.
  2. **Agents' own layers are trust-weighted.** An agent's update to the input and output layers comes from private
     data, so nobody can recompute it. Instead it is scored by how well it agrees in direction with a training
     step on a small **public seed set** (20 labelled examples per class, fixed at genesis). An update pointing the
     opposite way gets zero weight; this is the FLTrust approach. Updates are published only after everyone has
     committed to a hash of theirs, so nobody can copy another's update or react to it.
  3. **Vote.** Each agent measures how the candidate changes its own held-out loss, relative to the current model.
     The chain takes a **trimmed average** of these reports, ignoring the most extreme ones, and rejects the round
     only if it makes things more than 25% worse network-wide. Ordinary training steps often raise the loss a
     little and still make progress overall, so a stricter check throws away real learning. After a rejected
     round, everyone takes a smaller step.
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
| Can't tell if a server did the work correctly | Every answer carries a signed receipt. Every training step goes to all replicas and is compared, and 15% of inference hops are re-run on another replica. When 2 replicas agree against a third, the receipts prove it and the chain slashes the liar. In the demo, the lying server was caught at block 3 and agents then routed around it. | **Training: fixed. Inference: mitigated** (sampled, so high probability, not certainty). Needs an honest majority per slice and math that gives identical results on every machine. |
| Poisoned training updates from **servers** | Slice updates are recomputed by every replica from the same declared inputs, and the majority's result wins. The poisoning server was outvoted in **every round**: 100/100 on the toy task, 150/150 on MNIST and 42/42 on the real network. | **Fixed**, given an honest majority of each slice's replicas and identical math on each machine |
| Poisoned training updates from **agents** | Their updates come from private data, so they're trust-weighted against a public seed set. Updates that point the wrong way get zero weight. The cheating agent was excluded in 91–150 of the rounds, depending on the run. Honest agents with unusual data are sometimes down-weighted too. | **Mitigated**: it needs a small public labelled set. A saboteur can't damage what others teach, but if it is the *only* holder of some data (digits 4 and 9 in the MNIST demo), nobody else can teach the model that. |
| Speed of an internet-wide pipeline | Inference routes to the fastest trustworthy replica: **249 ms per hop** against 284 ms random, with 244 ms the best possible. Micro-batches pipeline through the slices: one job chunk took **1.8 s instead of 3.1 s**. | **Mitigated**: an internet pipeline is still much slower than one datacenter. That part is physics. |
| Incentives can be gamed | Joining needs a stake, which makes fake identities costly. Pay is per signed receipt, settled once per round. Training rewards go only to updates that were kept, in proportion to paid work. Cheaters are slashed and the reporter gets a bounty. | **Mitigated**: the prices and penalties are untested against real, motivated attackers. |
| Blockchain too slow for AI | Only hashes, Merkle roots and one batched payment per agent per round go on chain. | **Fixed** by design |
| Dishonest validator | The other 3 re-run each block and refuse forged ones (76 refused in the demo; the forger minted nothing). | **Fixed** for 1 bad validator in 4 (no full BFT view-change) |
| Servers can learn about your data | Raw data and final answers never leave the agent. Optional noise on activations, and optional differential privacy with an epsilon accountant. | **Partial**: servers still see hidden-layer activations. With 5 agents, strong DP costs too much accuracy (see below). Real fixes (secure computation, secure hardware) aren't built. |

### Measured results (`python simulate_mesh.py`, 100 rounds, about 25 s)

| Run | Test accuracy |
| --- | --- |
| Everyone honest (`--no-attackers`) | 93.4% |
| Lying server, poisoning server, cheating agent, forging validator and a crash, all at once | 94.2% |
| `--activation-noise 0.5` | 93.5% (but only 75% when the noise is also applied at inference) |
| `--privacy 0.3` (ε ≈ 2,560) | 88.8% |
| `--privacy 1.0` (ε ≈ 298) | 33.6%: unstable. It reached 74% at round 80, then collapsed. The 25% vote check lets very noisy steps through. |

Each agent alone (only 2 of the 4 classes) gets about 49%. Smaller ε means stronger privacy; useful values are under 10.
Getting there without losing accuracy takes thousands of participants, not 5.

### Running it as real, separate programs

```bash
python -m mesh.run local --rounds 40        # 21 OS processes on this machine, talking over TCP
python -m mesh.run init --dir net --host <ip>   # or: write a config, copy it to several machines,
python -m mesh.run node --dir net --name agent2 #     and start each node where you like
```

Each validator, server and agent is its own process listening on its own port. Validators take turns proposing
in fixed time slots, and a dead or forging proposer just loses its slot. Servers and agents read the chain like a
light client: they trust a state hash only if enough validators report it, then check the downloaded state
against that hash.

In a test run (40 rounds, 7 validators, about 5 minutes), test accuracy reached 89.7%:
- one server and one honest validator were killed with SIGKILL halfway through, and the network kept going
- the forging validator had all 24 of its blocks refused
- a hop took 3.2 ms over local sockets
- no node errors were logged

**Demo shortcut:** `init` writes every node's key seed into `nodes.json`. In real use, each machine should make its
own key and share only the public part.

### A bigger model: real handwritten digits (MNIST)

```bash
python simulate_mesh.py --dataset mnist                          # about 8 minutes
python -m mesh.run local --dataset mnist --rounds 30 --interval 4  # 21 real processes
```

Uses 70,000 real 28x28 digit images, downloaded once and checked against their published checksums. The model has
**1,982,986 parameters**: the agents' ends are 407,050, and each of the 3 slices is 525,312, held by 3 servers.
Each agent sees only 2 of the 10 digits.

Accuracy on the 10,000 test digits after 150 rounds:

| Run | Accuracy |
| --- | --- |
| Same model on one machine, no Mesh, fixed step size | 88.6% at the end (it touched 95.1% at one checkpoint) |
| Same, with the step size shrinking over the run | 91.2% |
| Mesh, everyone honest (average of the last 3 checkpoints, 3 seeds) | 88.9% (87.8% with a fixed step size) |
| Mesh, with all the attackers (1 run) | 82.1%: digits 0–3 and 5–8 at 89–99%; 4 and 9 (only the cheater has them) at 42% and 27% |

What this shows:
- **Size cost:** about 3.4 s of computer time and 280 MB of encrypted traffic per round. Over real sockets, with
  21 processes on a 4-core machine, a hop takes about 264 ms (3 ms with the toy model).
- **Earlier version:** 86.0% honest and 71.3% with attackers. There, the outlier check never caught the
  poisoning server in 2 million dimensions, and a vote that rejected any loss increase threw away half the rounds.
  Verified slice training, trust weighting and the relative 25% vote fixed both.
- **Single digits swing, even without Mesh.** With a fixed step size, one digit often lost more than 20 points
  in a single round; on one machine with no network and no attackers this happened 110 times after round 50.
  Each agent's gradient is clipped to a fixed size for privacy, and the output layer's gradients stay above that
  limit, so every step is the same size and that layer never settles. An earlier report that digit 8 had dropped
  to 37% was a snapshot taken during one of these dips: a rerun with the same settings ended with digit 8 at 96%.
  The step size now shrinks over the planned run (`lr_decay_rounds`, fixed at genesis so every node agrees). On
  one machine that cut the big dips from 110 to 21; in Mesh it cut them by about a third and raised the weakest
  digit from 70% to 77% on average.
- **What attackers still cost:** mostly the digits only the cheater holds.

### Not built yet

Validators joining or
leaving (the set is fixed at genesis) and full BFT consensus, defences against a majority of one
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
| `mesh/sim.py` | Builds a whole network in one process, with optional attackers; the public seed set |
| `mesh/datasets.py` | MNIST download (checksum-verified) and the per-class split |
| `mesh/transport.py`, `mesh/chain_net.py`, `mesh/run.py` | Real TCP networking, networked validators and light client, one-process-per-node runner |

---

## Swarm (the simple version)

Each peer trains on its own data, then swaps weights with a random peer and both move halfway toward each other.
They also swap peer lists, so everyone soon knows everyone. A shared key signs messages, and a limit caps how far
one peer can pull another. 8 peers that each see 2 of 4 classes reach about 92.5% together, against 48.6% alone.

```bash
python simulate.py --peers 8 --seconds 15 --secret k --kill 2
```

Each node exposes `GET /info`, `GET /model`, `POST /gossip` and `POST /predict` over HTTP.

---

## License

MIT; see [LICENSE](LICENSE). Contributions are accepted under the same license.
