# Security policy

## Status: research prototype — do not use with real value

Mesh is **unaudited research software**. Do not:

- attach real monetary value to its tokens, sell them, or promise they will be worth anything;
- train it on sensitive or personal data you can't afford to leak;
- run it anywhere a failure could hurt someone.

Its test-network tokens are play money. Turning them into something of value would also raise legal questions
(securities, money transmission, consumer protection) that need advice from a lawyer in your country, not code.

What Mesh does and doesn't defend against, and every assumption it relies on, is written up in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## Reporting a vulnerability

Breaking Mesh's defences is one of the most useful contributions to this project.

- **Ordinary attack reports** (forging a block, poisoning the model without being outvoted, stalling training,
  reading traffic, crashing nodes): open an issue with the
  [attack report template](https://github.com/Davy765/mesh-/issues/new?template=attack.md), ideally with a test that
  shows it. Because nothing of value runs on Mesh, public reports are fine and help everyone.
- **Anything that could harm people running a public test network** (for example, a way to run code on someone's
  machine through a crafted message): report it privately through GitHub's
  [private vulnerability reporting](https://github.com/Davy765/mesh-/security/advisories/new) instead of a public
  issue.

Please include what an attacker needs (which role, how many nodes, what access), what they can achieve, and how to
reproduce it.

## Supported versions

Only the `main` branch. There are no releases yet.
