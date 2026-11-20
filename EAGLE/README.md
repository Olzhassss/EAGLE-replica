# EAGLE experiments

EAGLE owns data and network preparation, policy learning, interactive inference,
and evaluation. It runs directly from the checkout on Python 3.13+.

```sh
uv sync --locked --project EAGLE
uv run --project EAGLE python -m EAGLE --help
```

Run commands from the repository root. `python -m EAGLE <command> --help` shows
each command's options. With `EAGLE/.venv` activated, omit the `uv run` prefix;
`make eagle ARGS='<command> ...'` uses the same environment. `pyproject.toml` and
`uv.lock` are the experiment environment's dependency definition and lockfile.

## Source map

| Location in `src/`                                     | Responsibility                                                                                                    |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `main.py`, `arguments.py`                              | Lazy command dispatch, shared flags, model names, and device selection                                            |
| `data_preparation/`                                    | Stream captures, pool one-second windows, rewrite replay packets, and generate synthetic/load/congestion datasets |
| `network_preparation/`                                 | Normalize switch graphs, partition domains, place controllers and hosts, allocate ports, and draw networks        |
| `model/`                                               | Neural networks, Feather embeddings, normalization, rewards, learner updates, consensus, and checkpoint I/O       |
| `training/`                                            | Episode stages, per-agent local/process workers, calibration, post-training, and resumable runs                   |
| `experiments/`                                         | Environment wiring, synchronized joint steps, trial execution, and run/event logs                                 |
| `communication/`                                       | HTTP controller requests and Unix socket traffic/supervisor commands                                              |
| `measurement/flows.py`                                 | TCP payload delivery and completion times, accounting for retransmissions                                         |
| `evaluation/`                                          | Policy comparisons, switch counts, embedding collection, and report plots                                         |
| `interactive.py`                                       | Apply a frozen policy to manually started traffic while displaying controller status                              |
| `network.py`, `datasets.py`                            | Load prepared inputs, validate dataset hosts/jobs, and compute content fingerprints                               |
| `config.py`, `settings.py`, `paths.py`, `artifacts.py` | Preparation constants, experiment settings, repository paths, and atomic output helpers                           |
| `../tests/`                                            | Input, numerical, training/resume, inference, and socket protocol regression tests                                |

Preparation defaults are module constants in `EAGLE.src.config`. Runtime learning
and reward parameters come from [settings presets](../settings/README.md).
Each controller slot owns a categorical action over its configured support-switch
combinations. Network preparation selects the requested switches and attaches
hosts to fixed switches outside those lists.

## Support-switch configuration

`config.SWITCHES_PER_DOMAIN` defaults to `1`. Override it during network generation
with `--support-switches K`, or provide one count per domain in controller order:

```sh
python -m EAGLE prepare-network random --nodes 30 --links 60 --hosts 9 --support-switches 2 --name two-per-domain
python -m EAGLE prepare-network random --nodes 30 --links 60 --hosts 9 --support-switches 2 1 0 --name mixed-supports
```

The resulting JSON contains an ordered `support_switches` list per controller.
Bit `i` of that controller's integer action enables list element `i`. For a list
`["s12", "s10"]`, the four actions are:

| Mask | s12 | s10 |
| ---- | --- | --- |
| 0    | off | off |
| 1    | on  | off |
| 2    | off | on  |
| 3    | on  | on  |

Counts can differ across domains; zero gives a fixed domain with the sole action
`0`. A domain with `k` support switches has `2^k` actor outputs. The joint action
space uses these per-domain cardinalities. Reward estimators receive the expanded
switch bits; critics and observations remain in controller order. All-on reset,
calibration, and baseline masks are derived from the generated network.

Selection preserves connectivity for every support-switch combination and leaves
host attachments on fixed switches. Impossible requests fail before output is
written. The fixed V0 graph cannot fit two support switches in every domain with
its existing host attachments; V3 and suitably sized random/Zoo graphs can.

Network JSON now uses schema version `2`; regenerate older network definitions.
Checkpoints record `support_counts` and reject incompatible target counts or
slot order. Changing support counts requires matching trained actors and reward
estimators. Older checkpoints without this metadata are rejected. Keep support
list order fixed within a run and use matching per-slot counts when transferring
a policy to another topology. Dataset schema version `1` is unchanged.

## Prepare and run

The default capture paths are `5gb-tcp-connection.pcap.gz` and `megalodon.pcapng`
at the repository root; `--dc` and `--mg` override them. Preparation creates a
new host-count directory and refuses to replace an existing one.

```sh
python -m EAGLE prepare-data default --hosts 9
python -m EAGLE prepare-network v0 --dataset datasets/9_hosts/train/dc_t0.json
python -m EAGLE prepare-data congestion --hosts 9 --network networks/v0.json
```

Use `--skip-pcaps` for synthetic-only preparation, or `--skip-train` for test
datasets only. Network preparation also supports `v3`, `random`, and `zoo`;
the [notebook](../summary.ipynb) contains the V0–V4 recipes.
Traffic manifests must use `schema_version: 1` and explicit `hosts` metadata. Regenerate
older datasets that lack these fields; addresses are not inferred from captures.

Start [Ryu](../ryu/README.md) and [Mininet](../mininet/README.md)
with the generated network before running experiments:

```sh
python -m EAGLE train --network v0.json --traffic synthetic --output dc_t0
python -m EAGLE train --network v0.json --traffic synthetic --output dc_t0 --resume
python -m EAGLE interactive --network v0.json --model dc_t0
python -m EAGLE eval-switches --network v0.json --steps 20
```

Training modes are `synthetic`, `pcaps`, and `curriculum`. Each starts with
reward-estimator pretraining; policy training then keeps that estimator frozen.
`--workers local` runs learners in the parent process; the default is one process
per controller slot. `--device` accepts `auto`, `cpu`, or an available CUDA device.
Post-training uses `--dataset NAME --from-checkpoint SOURCE --output NEW_NAME`.
Resume requires unchanged network, datasets, settings, seed, and training stages.

## Evaluation and notebooks

`eval-datasets` and `eval-topologies` compare three models with an all-on baseline.
`eval-switches` records enabled supporting switches over a fixed decision horizon.
`eval-traffic-embeddings` and `eval-topology-embeddings` collect all-on observations;
`eval-pruning` plots the three saved pruning reward histories. Topology evaluations
collect the currently running network per invocation and support `--plot-only`.
Outputs and run logs are stored under `models/` and `eval/`; see the
[repository overview](../README.md).

Open `summary.ipynb` using this environment. To register its kernel:

```sh
EAGLE/.venv/bin/python -m ipykernel install --user --name eagle --display-name EAGLE
```

Run `make test` from the root. The tests exercise preparation, CPU learning,
checkpoint resumption, mocked controller telemetry, and a local Unix socket.
