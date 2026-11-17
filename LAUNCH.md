# Launching Mesh on the open web

Mesh can launch as a **public test network**: anyone can watch it, run the demo, and join with their own computer,
using **play-money tokens**. It is not ready for real money or sensitive data, and nothing here changes that
(see [SECURITY.md](SECURITY.md)).

## What's ready

| Area | State |
| --- | --- |
| Code | Joining, leaving, faucet, status page, restarts, per-machine keys, HTTPS-capable transport |
| Hostile input | Malformed or forged messages and transactions are refused, never crash a node or stall the chain (tests/test_hardening.py) |
| Overload | Per-node concurrency cap, slow-client timeouts, per-address rate limits, bounded transaction queue, faucet daily cap |
| Web pages | Security headers (strict CSP, no framing), HTTPS via Caddy, rate limits |
| Supply chain | Bounded dependency versions, Dependabot updates, pip-audit and CodeQL in CI |
| Tests | Unit, fuzz-style hardening, 21-process network over TCP and HTTP with kill/restart, 3-machine rehearsal, test-network rehearsal, Docker build |
| Docs | [README](README.md), [deploy/TESTNET.md](deploy/TESTNET.md), [SECURITY.md](SECURITY.md), [threat model](docs/THREAT_MODEL.md), [CONTRIBUTING](CONTRIBUTING.md) |

## Launch steps

1. **Repository settings** (GitHub → Settings):
   - Security → turn on **Private vulnerability reporting**, **Dependabot alerts** and **Dependabot security
     updates**. Code scanning results from CodeQL then appear under the Security tab.
   - Branches → add a rule for `main`: require a pull request and require the **tests** checks to pass.
   - Turn on **Discussions** if you want a place for questions.
2. **Servers:** rent 4 small Linux servers for validators (1 vCPU, 1–2 GB is enough), ideally in different regions or
   providers, plus one for the status page and faucet (it can be one of the four). Point two domains at the status
   server, for example `status.yourdomain` and `faucet.yourdomain`.
3. **Set up** following [deploy/TESTNET.md](deploy/TESTNET.md) step by step: firewall, keys made on each server,
   genesis built from public cards, systemd services, Caddy for HTTPS.
4. **Smoke test before announcing:**
   - the status page loads over HTTPS and shows 4 validators and a rising block height;
   - from a computer outside your servers, `python -m mesh.run join ... --role agent --index 0` gets tokens,
     registers, and appears on the status page within a minute;
   - training rounds are being accepted.
5. **Announce** (README "Get involved" box, plus wherever people interested in decentralized AI gather). Say plainly:
   it's a research test network, tokens have no value, don't bring sensitive data, and attack reports are welcome.

## Running it

- **Watch** the status page daily at first: block height rising, rounds accepted, no unexpected slashes, validators
  all present.
- **Updates:** merge Dependabot pull requests once CI is green; then on each server `git pull` and
  `sudo systemctl restart mesh-node@NAME` one validator at a time (the network keeps going while one restarts).
- **Backups:** `/var/lib/mesh/net/data/` holds each validator's saved blocks; the keys in `/var/lib/mesh/net/keys/`
  must be backed up somewhere safe and never shared.
- **If a validator's key leaks:** have it leave (`python -m mesh.run leave ...`), make a new key, and join again as a
  new validator.
- **If something breaks badly:** it's a test network. Stop the services, fix, and if needed start over from a fresh
  genesis; nobody loses anything real.
- **Faucet abuse:** lower `--per-day` or `--amount`, or stop the faucet for a while.

## Never

- Sell the tokens, promise they'll be worth something, or connect them to real money.
- Let people train on personal or confidential data.
- Remove the research-prototype warning from the README or the status page.
