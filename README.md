# EAGLE replica

EAGLE trains cooperative SDN agents to balance delivered throughput and modeled power. The implementation uses native PyTorch, FeatherGraph observations, and consensus with dynamic pruning enabled by default.

[Documentation.md](Documentation.md) covers preparation, configuration, model artifacts, metrics, and all five evaluations. [pipeline.ipynb](pipeline.ipynb) walks through the complete workflow with visualizations and manual topology changes.

| Component | Runtime | Guide |
| --- | --- | --- |
| Network, dataset, training, and evaluation tools | Python 3.13 in `model/.venv` | [Root documentation](Documentation.md) |
| OpenFlow controllers | Python 3.7 in `ryu_controller/.venv` | [Ryu](ryu_controller/README.md) |
| Emulation and traffic | System Python 3.10+ with Mininet and Open vSwitch | [Mininet](mininet_nets/README.md) |

## Prepare inputs

Run from the repository root after installing the dependencies in the guides. Dataset generation uses the local Digital Corpora and Megalodon captures.

```sh
model/.venv/bin/python -m data_preparation default --hosts 9 --min-decisions 20
model/.venv/bin/python -m network_preparation v0 --dataset datasets/9_hosts/train/dc_t0.json
model/.venv/bin/python -m network_preparation v3 --dataset datasets/9_hosts/train/dc_t0.json
model/.venv/bin/python -m data_preparation congestion --hosts 9 --network networks/v0.json
```

V0 uses the checked-in source graph. V3 extends it to 16 switches and 22 inter-switch links. Both take host identities and addresses from the dataset. Random and Topology Zoo generators provide V1, V2, and V4 as described in the root documentation.

## Train

Start Ryu first, then Mininet, in separate terminals with the same generated network.

```sh
ryu_controller/.venv/bin/python ryu_controller/launch_controllers.py --scenario networks/v0.json
sudo python3 mininet_nets/launch_topology.py --scenario networks/v0.json
```

```sh
model/.venv/bin/python -m model.train --network v0.json --traffic synthetic --output dc_t0
model/.venv/bin/python -m model.train --network v0.json --traffic pcaps --output dc_pcaps
model/.venv/bin/python -m model.train --network v0.json --traffic curriculum --output dc_full
```

Training resolves datasets from the network's host count. It calibrates observation normalization on source training traffic, pretrains the reward estimator, and freezes it during policy learning. Output includes `models/<name>`, `models/<name>.csv`, `models/<name>.png`, and `models/checkpoints/<name>.pth`. Use `--settings` for a JSON configuration and `--resume` to continue the same named model.

## Evaluate

```sh
model/.venv/bin/python -m model.eval_topology_embeddings --network v0.json --model dc_t0
model/.venv/bin/python -m model.eval_traffic_embeddings --model dc_t0
model/.venv/bin/python -m model.eval_datasets --models dc_t0 dc_pcaps dc_full
model/.venv/bin/python -m model.eval_topologies --network v0.json --models dc_t0 dc_pcaps dc_full
model/.venv/bin/python -m model.eval_pruning
```

For the two topology evaluations, restart Ryu and Mininet manually and repeat collection for each of `v0.json` through `v4.json`. Collection generates the aggregate plots once all five results are available. `--plot-only` redraws them from saved results. Evaluation figures are saved under `eval/`.
