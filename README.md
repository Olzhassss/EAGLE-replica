# EAGLE replica

This project is build largely inspired by Monaco et al., _Adaptive SDN Autoscaling via Generalizable Multi-Agent Reinforcement Learning With EAGLE_, IEEE TNSM 2026, DOI `10.1109/TNSM.2026.3717361`. See the PDF for the whole paper.

Initially forked form the [original authors' repository](https://github.com/dahara98/EAGLE), the codebase has been rewritten to partially satisfy the original methodology with modern Python and prepared evaluation.

EAGLE trains cooperative SDN agents to balance delivered throughput and modeled power. The implementation uses native PyTorch, FeatherGraph observations, and consensus with dynamic pruning enabled by default.

[Documentation.md](Documentation.md) covers preparation, configuration, model artifacts, metrics, and all six evaluations. [pipeline.ipynb](pipeline.ipynb) walks through the complete workflow with visualizations and manual topology changes.

| Component                                        | Runtime                                           | Guide                                  |
| ------------------------------------------------ | ------------------------------------------------- | -------------------------------------- |
| Network, dataset, training, and evaluation tools | Python 3.13 in `model/.venv`                      | [Root documentation](Documentation.md) |
| OpenFlow controllers                             | Python 3.7 in `ryu_controller/.venv`              | [Ryu](ryu_controller/README.md)        |
| Emulation and traffic                            | System Python 3.10+ with Mininet and Open vSwitch | [Mininet](mininet_nets/README.md)      |

## Prepare inputs

Run from the repository root after installing the dependencies in the guides. Dataset generation uses the local Digital Corpora and Megalodon captures.

```sh
model/.venv/bin/python -m data_preparation default --hosts 9 --min-decisions 20
model/.venv/bin/python -m network_preparation v0 --dataset datasets/9_hosts/train/dc_t0.json
model/.venv/bin/python -m network_preparation v3 --dataset datasets/9_hosts/train/dc_t0.json
model/.venv/bin/python -m data_preparation congestion --hosts 9 --network networks/v0.json
```

V0 uses the checked-in source graph. V3 extends it to 16 switches and 22 inter-switch links. Both take host identities and addresses from the dataset. Random and Topology Zoo generators provide V1, V2, and V4 as described in the root documentation.

`Config.TRAFFIC_SCALE_FACTOR` in `config.py` stretches traffic timing. Its default `None` uses the host count. `min_decisions` applies to the final stretched duration. Repetition fills a shortage after scaling. With nine hosts and five-second decisions, 12 source seconds become 108 seconds, providing 21 decisions and satisfying a minimum of 20. Synthetic jobs use the same longer schedule and reduced rates.

Use `--skip-pcaps` to generate synthetic datasets and avoid packet spooling. Add `--skip-train` to prepare test data only. The latter option also applies to congestion preparation.

```sh
model/.venv/bin/python -m data_preparation default --hosts 50 --skip-pcaps --skip-train
```

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

The notebook also post-trains all three models on `dc_t5` before evaluation. Use `--from-checkpoint <source_name> --dataset dc_t5 --split test --output <new_name> --episodes <additional_episodes>` to restore full training state and continue policy learning with the reward estimator frozen. Post-training saves separate models and reward curves. See `Documentation.md` for resumption details.

## Interactive simulation

Start Ryu with `networks/v0.json` as above. In a second terminal, open Mininet's CLI; in a third, attach a trained policy:

```sh
sudo python3 mininet_nets/launch_topology.py --scenario networks/v0.json --cli
model/.venv/bin/python -m model.interactive --network v0.json --model dc_full_t5
```

Trigger traffic manually at `mininet>` while the policy terminal prints `s12 (c0)`, `s11 (c1)`, and `s13 (c2)` with their observed ON/OFF forwarding states every second. Decisions retain the five-second default cadence. See [the interactive workflow](Documentation.md#interactive-simulation-with-manual-traffic) for traffic commands, timing options, and shutdown behavior.

## Evaluate

```sh
model/.venv/bin/python -m model.eval_topology_embeddings --network v0.json --model dc_t0
model/.venv/bin/python -m model.eval_traffic_embeddings --model dc_t0
model/.venv/bin/python -m model.eval_datasets --models dc_t0 dc_pcaps dc_full
model/.venv/bin/python -m model.eval_topologies --network v0.json --models dc_t0 dc_pcaps dc_full
model/.venv/bin/python -m model.eval_pruning
model/.venv/bin/python -m model.eval_switches --network v0.json --steps 20
```

For the two topology evaluations, restart Ryu and Mininet manually and repeat collection for each of `v0.json` through `v4.json`. Collection generates the aggregate plots once all five results are available. `--plot-only` redraws them from saved results. Evaluation figures are saved under `eval/`.

The switch-count evaluation runs one episode each for the three models and the all-on baseline. It plots the sum of binary actions after each decision and includes the final counts in the legend. DC synthetic T0 is the default test dataset. Use `--dataset` to select another prepared test manifest by name, and `--models` to select three inference files. Outputs are `eval/switches_<network>.png` and the corresponding JSON measurements.
