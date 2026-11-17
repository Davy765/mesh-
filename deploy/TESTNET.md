# Running a public Mesh test network

A test network runs all the time on a few cloud servers, and anyone can join with their own computer as a server or
agent. **Its tokens are play money with no value, and Mesh is unaudited research software:** don't attach value to the
tokens and don't let people train on sensitive data. See [SECURITY.md](../SECURITY.md).

## What you need

- **4 small cloud servers** for validators (1 vCPU, 1–2 GB RAM is enough for the toy task), ideally from different
  providers or regions. 4 is the minimum: the chain then keeps working with 1 validator down or dishonest.
- One of them (or a fifth) also runs the **status page** and the **faucet**.
- Each server needs a public IP, Python 3.10+ (or Docker), and **port 9100** open for the node.
- The status/faucet server also needs **ports 80 and 443** open (HTTPS, below) and two domain names pointed at it,
  for example `status.yourdomain` and `faucet.yourdomain`. Keep 8080 and 8090 closed to the internet: Caddy serves
  them over HTTPS.
- A firewall that allows only those ports, for example:
  `sudo ufw default deny incoming && sudo ufw allow 22/tcp && sudo ufw allow 9100/tcp && sudo ufw enable`
  (and `sudo ufw allow 80,443/tcp` on the status/faucet server).

## 1. On each validator server

```bash
git clone https://github.com/Davy765/mesh- /opt/mesh && cd /opt/mesh
pip install -r requirements.txt
python -m mesh.run keygen --keys /var/lib/mesh/net/keys --name validator0 --role validator \
    --endpoint PUBLIC_IP:9100 --listen 0.0.0.0:9100 > validator0.card.json
```

Use `validator1`, `validator2`, `validator3` on the others. The private key stays in `/var/lib/mesh/net/keys`;
only the `.card.json` file (public keys and address) gets shared.

## 2. On the server that will run the faucet

```bash
python -c "from mesh.crypto import Identity; i=Identity.generate(); i.save('faucet.key'); print(i.address)"
```

Keep `faucet.key` private; note the printed address.

## 3. Build the network config (anywhere)

Copy the four `.card.json` files to one machine, then:

```bash
python -m mesh.run genesis --dir net --cards validator*.card.json --faucet-address FAUCET_ADDRESS
```

This writes `net/genesis.json` and `net/nodes.json` (no secrets). Copy both to `/var/lib/mesh/net/` on every
validator server.

## 4. Start everything

On each validator server, with the systemd unit (restarts automatically):

```bash
sudo useradd --system mesh && sudo chown -R mesh /var/lib/mesh
sudo cp deploy/mesh-node@.service /etc/systemd/system/
sudo systemctl enable --now mesh-node@validator0      # its own name on each server
```

On the status/faucet server:

```bash
python -m mesh.run status --genesis /var/lib/mesh/net/genesis.json --port 8080 \
    --public-url https://status.yourdomain/genesis.json &
python -m mesh.run faucet --genesis /var/lib/mesh/net/genesis.json --key faucet.key --port 8090 --trust-proxy &
```

(Or run each as a systemd service the same way.) Then put them behind HTTPS with [Caddy](https://caddyserver.com):
edit the two domains in [deploy/Caddyfile](Caddyfile) and run `sudo caddy run --config deploy/Caddyfile` (or install
it as a service). Caddy gets and renews certificates by itself. `--trust-proxy` makes the faucet rate-limit by the
visitor's real address, which Caddy forwards; only use it behind a proxy. Open `https://status.yourdomain` to watch
the network.

With Docker instead of a Python install, build once with `docker build -t mesh .`, then run each node as
`docker run -d --restart unless-stopped -v /var/lib/mesh:/var/lib/mesh -p 9100:9100 mesh node --dir /var/lib/mesh/net --name validator0`.

## 5. Tell people how to join

Anyone with a computer that others can reach (a public IP or an open port) can join as a slice server or an agent:

```bash
git clone https://github.com/Davy765/mesh- && cd mesh- && pip install -r requirements.txt
python -m mesh.run join --genesis https://status.yourdomain/genesis.json --faucet https://faucet.yourdomain \
    --name YOUR_NAME --role server --piece slice0 --endpoint YOUR_PUBLIC_IP:9100 --listen 0.0.0.0:9100
```

(`--role agent --index 0` to train as an agent on the demo data.) `join` creates the key locally, asks the faucet for
test tokens, registers, downloads the current weights and starts taking part. To become a validator, use
`--role validator`: it joins the set at the next epoch boundary (every 50 blocks), after staking 1,000 test tokens.

## Rehearse it on one machine first

```bash
python -m mesh.run local --testnet --rounds 20
```

Runs the whole thing locally with a faucet and status page; partway through, a new agent and a new validator join
from scratch.

## Keeping it healthy

- Watch the status page: rounds accepted, validators, slashes.
- Validator data lives in `/var/lib/mesh/net/data/` (saved blocks); back it up if you care about the history.
- If a validator server dies, restart it: it replays its saved blocks and catches up from the others.
- To retire a validator: `python -m mesh.run leave --genesis net/genesis.json --key keys/NAME.key`. It leaves at the
  next epoch boundary and gets its stake back; the set never drops below 4.
