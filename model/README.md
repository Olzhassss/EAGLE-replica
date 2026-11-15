# EAGLE learning and experiment project

Use a separate modern Python environment. These commands use Python 3.11 and run from the repository root.

```sh
python3.11 -m venv model/.venv
model/.venv/bin/python -m pip install -r model/requirements.txt
```

Start the [Ryu](../ryu_controller/README.md) and [Mininet](../mininet_nets/README.md) projects with the same scenario first. Set `MININET_SOCKET` to the socket emitted by the Mininet launcher.

```sh
model/.venv/bin/python -m model.train --mininet-socket "$MININET_SOCKET" \
  --traffic synthetic --episodes 3 --output runs/train-v0

model/.venv/bin/python -m model.evaluate --mininet-socket "$MININET_SOCKET" \
  --traffic synthetic --checkpoint runs/train-v0/checkpoint.pth --output runs/eval-v0
```

## Entry points and files

| Entry point            | Use                                                                                                                                                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.train`          | Train per-agent actor, critic, and reward estimator models. Use `--workers local` for an in-process run or the default `process` for spawned workers. JSON settings can define pretraining, frozen-estimator, and simultaneous learning stages. |
| `model.evaluate`       | Replay a frozen checkpoint, run a fixed action vector, or call an external baseline. Use `--policy module:function` for a callable taking `(observation, snapshots)` and returning one integer binary action per logical slot.                  |
| `model.sweep`          | Repeat evaluation cells with common seeds and save the outcome of each subprocess.                                                                                                                                                              |
| `model.prepare_traces` | Prepare Mininet replay PCAPs and their scenario from the Digital Corpora gzip or another Ethernet libpcap file. See the [trace workflow](../scenarios/README.md#run-the-digital-corpora-capture).                                               |
| `model.report`         | Compute run-level metrics, matched-baseline comparisons and 90% intervals. Options produce plots and t-SNE coordinates from recorded embeddings.                                                                                                |
| `partitioning`         | Partition an offline graph and calculate controller locations.                                                                                                                                                                                  |

`controller_client.py` owns HTTP readiness, acknowledgments, and retries. `joint_environment.py` owns trial reset, joint action timing, and coherent sample selection. `observations.py` and `reward.py` implement numerical preprocessing and reward calculation. The actor-critic, optimizer and consensus code live in `actor_critic.py`, `learner.py`, `consensus.py`, and `workers.py`.

`global_obs` is a float32 vector in scenario controller order. A local Feather fingerprint has `4 × order × eval_points` features. V0 has a 12-element vector with the default `g=1`, `e=1`. Training batches preserve that slot order in actions and rewards. A trial horizon produces a truncation and retains bootstrapping for continuing-state value estimates.

## Checkpoints and inputs

A run writes settings and provenance to `run.json` and controller snapshots, raw embeddings, actions, optimizer information, timing, and flow observations to `events.jsonl`. Training saves `checkpoint.pth` at each episode boundary. For `--resume`, keep the original settings, stage schedule, scenario, and seed, and use a new output directory. The saved checkpoint carries the optimizer, running reward, pruning round, and per-agent sampling state. Legacy combined checkpoint dictionaries can be loaded with `--checkpoint`. Use repeated `--legacy-agent ID=PATH` options for old multi-file checkpoints keyed by the original slot indices.

For new source normalization, record fixed-policy source runs with `--identity-normalization`, then pass their events to training with `--normalize-from`. That fitted state travels in the checkpoint and is reused for target evaluation. Use an identical observation view and controller slot order when loading the checkpoint on a new topology. Use `--replay-from events.jsonl` for offline training on recorded transitions, with a settings profile and source embeddings that match the recording. The runner re-encodes the saved raw embeddings with the selected normalization.

See [study inputs](../scenarios/README.md#published-protocol) for the values that require original data. For a remote testbed, `--traffic external` uses the scenario controllers without a local Mininet socket. Supply receiver measurements as JSON lines with `--flow-records`. Each record carries `episode` (optional, defaults to 1), `offered_bytes`, `received_bytes`, and `fct_seconds` (null for incomplete flows). Optional fields include `matched_packets`, `packet_delay_sum_seconds`, and `first_byte_seconds`.

## Repeat and compare trials

A matrix JSON contains `schema_version: 1`, optionally `seeds` (defaults to 0–9), and `cells`. Each cell supplies a path-safe `condition`, `method`, and an `arguments` list for `model.evaluate`. Use a condition that identifies the topology, traffic, resource and failure settings, and share it with the static baseline. Start the required emulation sessions before running the matrix. For latency trials set 25 distinct seeds.

```sh
model/.venv/bin/python -m model.sweep matrix.json --output runs/matrix
MPLBACKEND=Agg model/.venv/bin/python -m model.report runs/matrix/*/*/*/ \
  --baseline static --output runs/report.json --plots runs/figures
```

Reports pair baseline runs by condition and seed and retain incomplete-flow counts and failed runs. The chosen FCT gain normalization scope is recorded with the efficiency denominator. FCT uses the source start and receiver completion of the offered payload. For PCAP traces, packet delay matches exact captured payload segments and reports how many matched. Keep power in watts, time-integrated energy in joules, and carbon in grams distinct. Power and carbon require coefficients from the chosen experiment.

For the short smoke run, use the synthetic fixed-action trial in the root guide. As a follow-up, train one episode and reopen the produced checkpoint using a separate evaluation run.
