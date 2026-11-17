# Mesh threat model

A starting point for anyone reviewing or auditing Mesh. It lists what Mesh protects, against whom, which assumptions
it depends on, how each defence was tested, and what is known to be missing. **Mesh is a research prototype and not
fit for anything of real value** (see [SECURITY.md](../SECURITY.md)).

## What is being protected

| Asset | Why it matters |
| --- | --- |
| The model's weights | Poisoned weights give wrong answers to everyone who uses the model. |
| Agents' training data | It never leaves its owner; leaks would break the main promise of the design. |
| The ledger | Balances, stakes, reputation, the agreed model version and the validator set all live here. |
| Answers to inference and job requests | Users pay for them and act on them. |
| Availability | An attacker who can stall training or the chain can make the network useless. |
| Node operators' machines | Anyone running a node accepts network messages from strangers. |

## Who might attack

- **A dishonest server** holding a slice: lies about results, poisons updates, withholds data.
- **A dishonest agent:** poisons its updates, lies in votes, fakes job results, withholds training inputs.
- **A dishonest validator:** forges blocks, mints tokens, stalls the chain, equivocates.
- **Colluding groups** of the above.
- **A network attacker:** eavesdrops, tampers with or replays messages, impersonates nodes.
- **A Sybil attacker:** creates many identities to gain a majority.

## Assumptions the defences rely on

1. **Fewer than a third of validators are faulty** (n ≥ 3f + 1, quorum n − f). More than that can forge or stall the
   chain.
2. **Each slice has an honest majority of replicas.** With 3 replicas, 2 colluding servers can poison that slice
   undetected.
3. **Honest replicas compute the same result to within the update grid** (2^-24). Tested against simulated
   differences of 1e-12, not yet on physically different hardware (issue #6).
4. **A small public labelled seed set exists** and is representative; agents' trust scores are measured against it.
5. **Ed25519, X25519, AES-256-GCM and SHA-256 are secure,** as implemented by the `cryptography` library.
6. **Clocks are roughly synchronised** for the time-slot leader rotation (a wrong clock costs liveness, not safety).
7. **Stake makes identities costly.** The values (100 for nodes, 1,000 for validators) are arbitrary and untested
   against a motivated attacker.

## Defences and how far they've been tested

| Threat | Defence | Evidence | Status |
| --- | --- | --- | --- |
| Forged or minted blocks | Every validator re-executes each block; a quorum must sign | Forging validator had every proposal refused in all demo runs | Holds under assumption 1 |
| Lying servers (inference) | Signed receipts; 15% of hops re-run on another replica; 2-of-3 proof slashes the liar | Liar caught within a few blocks in demos | Probabilistic, not certain |
| Lying servers (training) | Every training step goes to all replicas and is compared | Liar caught at block 3 | Holds under assumption 2 |
| Poisoned server updates | Every replica recomputes updates from on-chain-declared inputs; majority hash wins | Poisoner outvoted every round (100/100, 150/150, 42/42 over TCP) | Holds under assumptions 2–3 |
| Poisoned agent updates | Trust weighting against the public seed set; size caps; network-wide vote | Cheater excluded in nearly every round | Mitigated; needs assumption 4; can't recover knowledge only the cheater holds |
| Withholding inputs or salts | Replicas fetch missed data from each other, checked against on-chain fingerprints | Test: the attack stalls training without the defence, not with it | Fixed (tested) |
| Eavesdropping, tampering, replay | Per-pair AES-256-GCM, signatures, nonces | Eavesdropper read nothing; tampering and replay tests | Content protected; traffic metadata visible |
| Impersonation | Keys registered on chain; unknown senders refused | Test | Holds |
| Fake job results | Signed claims, poster spot checks, 2-witness disputes | Cheater slashed in demos | Probabilistic (spot checks) |
| Key theft from config | Each machine generates its own key; config holds public keys only | Test checks no private material in genesis | Fixed |
| Node restart / crash | Saved blocks replayed and re-verified; weights resynced and hash-checked | Kill-and-restart tests over TCP and HTTP | Holds |
| Leaks of training data | Data stays local; optional DP noise and activation noise | ε ≈ 298 at noise 1.0 with 5 agents | **Weak**: servers see hidden activations |

## Known gaps (not defended)

- **Collusion of a slice majority** (assumption 2). Mitigation would need more replicas or random audits.
- **Traffic analysis:** who talks to whom and message sizes are visible.
- **Privacy at small scale:** meaningful differential privacy needs far more agents; activation privacy would need
  secure computation or trusted hardware.
- **Consensus:** time-slot rotation is not a full BFT protocol (no formal view change; equivocation is prevented by
  quorum signatures in practice, not by a proof).
- **Economics:** prices, stakes, slashing and rewards have never been simulated against rational attackers.
- **Denial of service:** nodes cap open connections (1,024) and requests being processed (64), drop slow senders
  after 15 s, rate-limit each address, and bound the transaction queue; malformed input is refused without crashing
  (tests/test_hardening.py). A large distributed flood can still exhaust a node's network or CPU: put public endpoints
  behind a provider firewall or DDoS protection if that matters.
- **Faucet:** it's a centralised convenience for test networks, limited per address and IP, and easy to drain with
  many addresses.
- **Code quality for security:** fuzz-style tests for message and transaction handling, CodeQL scanning and
  pip-audit in CI, but no coverage-guided fuzzing and no independent review.

## Before Mesh could handle anything of real value

None of this is done. All of it would be needed:

1. An **independent security audit** of the protocol and the code, using this document as the starting point.
2. A **formally specified BFT consensus** (or an established one) with proofs or a model-checked spec.
3. **Economic analysis and simulation** of incentives against rational and colluding attackers.
4. **Stronger privacy:** secure aggregation or trusted execution for activations; DP at a scale where ε is
   meaningful.
5. **Hardening:** rate limiting, fuzzing of every message parser, resource limits, dependency pinning and
   supply-chain checks, key management (hardware keys for validators).
6. **Operations:** monitoring, incident response, upgrade and key-rotation procedures.
7. **Legal advice** on what the tokens are and what obligations running the network creates.
