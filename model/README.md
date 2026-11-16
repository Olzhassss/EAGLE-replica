# EAGLE learning and experiment project

Use a separate modern Python environment. These commands use Python 3.11 and run from the repository root.

```sh
python3.11 -m venv model/.venv
model/.venv/bin/python -m pip install -r model/requirements.txt
```

Start the [Ryu](../ryu_controller/README.md) and [Mininet](../mininet_nets/README.md) projects with the same scenario first.

Follow [pipeline.md](../pipeline.md) to collect source calibration, train, and evaluate. Local trials discover the launcher addresses in `mininet_nets/.socket` and the repository-root `.socket`. The corresponding CLI flags override these addresses.

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

Training and checkpoint evaluation default to `--device auto`, which selects CUDA when available to PyTorch and CPU otherwise. Use `--device cuda` to require a GPU, `--device cuda:1` to select the second visible GPU, or `--device cpu` to force CPU. All training workers use the selected device. The run manifest records the device, and checkpoints can be loaded and resumed on either CPU or CUDA.

`global_obs` is a float32 vector in scenario controller order. A local Feather fingerprint has `4 × order × eval_points` features. V0 has a 12-element vector with the default `g=1`, `e=1`. Training batches preserve that slot order in actions and rewards. A trial horizon produces a truncation and retains bootstrapping for continuing-state value estimates.

## Run settings

[`default.json`](../settings/default.json) is the study-oriented example used by training and evaluation. It uses five-second decisions, source-fitted normalization, throughput and power reward, and the current-state binary-action estimator. Its power coefficients, normalization references, batch and timing inputs, and single online stage are local example choices. Use the source calibration commands in the pipeline before training.

[`study.json`](../settings/study.json) specifies the reported five-second decision interval, Feather dimensions, actor and critic learning rates, reward coefficient, dynamic pruning endpoints, and five curriculum stages after estimator pretraining. Copy it and fill the null fields using recovered experiment values. Source calibration uses `model.evaluate --identity-normalization` followed by `model.train --normalize-from`.

Each training stage can specify `active_agents` as a subset of logical slots participating in learner updates. The trainer still produces an action for every slot. Optionally supply `learning.mixing_weights` as one map of peer ID to weight per receiving slot. Each row is renormalized over active contributors. The default mixes all active agents equally.

Each settings profile can specify a resource mode (`forwarding`, `interfaces`, or `ovs`), and a list of failure events. Events run before their zero-based step `step`. Specify `kind: "link"`, `source`, `target` and boolean `up` for a link event. Use `kind: "controller"`, `controller_id` and boolean `up` for a controller outage or restore. For modeled carbon, specify `carbon.intensity_grams_per_kwh` for every active link endpoint and switch in a power-enabled configuration.

## Checkpoints and inputs

A run writes settings and provenance to `run.json` and controller snapshots, raw embeddings, actions, optimizer information, timing, and flow observations to `events.jsonl`. Training saves `checkpoint.pth` at each episode boundary. For `--resume`, keep the original settings, stage schedule, scenario, and seed, and use a new output directory. The saved checkpoint carries the optimizer, running reward, pruning round, and per-agent sampling state. Use `--checkpoint` to evaluate a native EAGLE training checkpoint.

For new source normalization, record fixed-policy source runs with `--identity-normalization`, then pass their events to training with `--normalize-from`. That fitted state travels in the checkpoint and is reused for target evaluation. Use the same controller slot order when loading the checkpoint on a new topology. Use `--replay-from events.jsonl` for offline training on recorded transitions, with a settings profile and source embeddings that match the recording. The runner re-encodes the saved raw embeddings with the selected normalization.

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
