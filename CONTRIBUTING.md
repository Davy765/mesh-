# Contributing to Swarm & Mesh

Thanks for wanting to help. Mesh is a research prototype of decentralized AI: one model split across many
servers, trained by agents who keep their data private, coordinated by a small blockchain. Plenty of it is
unsolved, and fresh eyes are useful, whether you know distributed systems, cryptography, machine learning or
none of those yet.

## Where to start

- **Issues labelled [`good first issue`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** are small and self-contained.
- **Issues labelled [`help wanted`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)** are bigger open problems. Many are research questions; a careful experiment with an honest write-up is a real contribution, even if it shows an idea doesn't work.
- The README's **"problems and what each fix achieves"** table lists what is fixed, mitigated or still open.
- Have an idea that isn't an issue yet? Open one first so we can talk it through before you write a lot of code.

## Setting up

You need Python 3.10 or newer.

```bash
git clone https://github.com/Davy765/mesh- && cd mesh-
pip install -r requirements.txt            # numpy and cryptography
python -m unittest discover -s tests       # about 15 seconds; everything should pass
python simulate_mesh.py                    # the whole system with attackers, about 25 seconds
```

`python simulate_mesh.py --dataset mnist` downloads MNIST (about 11 MB, checksum-verified) and takes about
8 minutes. `python -m mesh.run local` runs every node as its own process over TCP.

## Making a change

1. Fork the repo and create a branch from `main`.
2. Keep the change focused: one fix or feature per pull request.
3. Add or update a test in `tests/` for anything that changes behaviour.
4. Run the tests: `python -m unittest discover -s tests`. GitHub runs them too on every pull request.
5. If you change results (accuracy, attack handling, speed), say **how you measured** in the pull request:
   which command, how many rounds, how many seeds. One run can mislead (see the digit-8 story in the README),
   so prefer several seeds or say plainly that it's one run.
6. Open a pull request and fill in the template.

## Style

- Plain Python, numpy and `cryptography`; please don't add heavy dependencies without discussing it first.
- Explain *why* in comments where the reason isn't obvious, especially for anything security-related.
- Report what you measured honestly, including what got worse. A clearly stated limitation is far more useful
  than an overclaimed result.

## Security

Mesh is a prototype and not safe to use with real money or sensitive data. If you find a way to break one of
its defences (forge a block, poison the model without being outvoted, read someone's traffic), that's a great
contribution: open an issue describing the attack, ideally with a test that demonstrates it.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
