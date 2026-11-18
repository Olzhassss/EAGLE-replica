# EAGLE pipeline

The source reference is Monaco et al., _Adaptive SDN Autoscaling via Generalizable Multi-Agent Reinforcement Learning With EAGLE_, IEEE TNSM 2026, DOI `10.1109/TNSM.2026.3717361`. The local PDF is retained as the research reference. Network variants, split windows, power coefficients, and training budgets in this checkout define a reproducible local experiment.

## Environments and layout

Run commands from the repository root. Create the learning environment with Python 3.13.

```sh
python3.13 -m venv model/.venv
model/.venv/bin/python -m pip install -r model/requirements.txt
```

For the notebook, install Jupyter and a kernel in that environment.

```sh
model/.venv/bin/python -m pip install jupyterlab ipykernel
model/.venv/bin/python -m ipykernel install --user --name eagle --display-name EAGLE
model/.venv/bin/python -m jupyterlab pipeline.ipynb
```

Use the [Ryu guide](ryu_controller/README.md) for its Python 3.7 dependencies and the [Mininet guide](mininet_nets/README.md) for system dependencies. The original Python 3.7 application is available in Git revision `2577b26`.

| Location                    | Contents                                                                           |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `data_preparation/`         | Capture analysis, split windows, address rewriting, and dataset generation         |
| `network_preparation/`      | Source V0, V3 extension, random and Zoo graphs, partitioning, and placement        |
| `networks/`                 | Generated topology JSON and SVG files                                              |
| `datasets/<n>_hosts/train/` | Source training manifests and PCAP files                                           |
| `datasets/<n>_hosts/test/`  | Held-out source, Megalodon, load, and congestion manifests                         |
| `settings/`                 | Default settings and pruning overrides                                             |
| `model/`                    | Learners, runtime clients, training, measurement, and six evaluation entry points |
| `models/`                   | Named inference files and training reward CSV/PNG files                            |
| `models/checkpoints/`       | Latest resumable checkpoints and per-model run logs                                |
| `eval/`                     | Evaluation figures, reward CSVs, measurements, and projection coordinates          |

Generated networks, datasets, models, and evaluation results are Git-ignored.

## Dataset preparation

The default capture names are `5gb-tcp-connection.pcap.gz` and `megalodon.pcapng`, with Megalodon's Content Server capture used as input. Override them with `--dc` and `--mg`. Capture analysis is available separately.

```sh
model/.venv/bin/python -m data_preparation.analyze_trace
model/.venv/bin/python -m data_preparation.analyze_trace --mg
model/.venv/bin/python -m data_preparation default --hosts 9
```

Generation writes a fresh `datasets/9_hosts` directory. Move existing output aside before rebuilding it. `--output-dir` selects another parent directory during preparation. The training and evaluation pipeline resolves its inputs under the repository's `datasets/` directory.

Digital Corpora is divided into complete one-second windows and split 80/20 using the provided seed. Each ordered host pair receives a shuffled pool from the appropriate split. The pool is exhausted before windows are reused. Train and test source windows remain disjoint. Megalodon is evaluation-only, with the selected `192.168.1.1` to `172.28.190.238` pair tagged separately from other traffic before pooling.

For `P = n(n-1)` ordered pairs, split size `W` one-second windows, time scale `X`, and decision interval `tau`, each pair can receive `A = floor(W / P)` unique source windows. These provide `floor(A * X / tau)` decisions after scaling. If that meets `min_decisions`, the available windows are used directly. Otherwise, each pair receives `ceil(min_decisions * tau / X)` source windows, repeating only after exhausting the split. Whole-window rounding ensures the final duration provides at least the requested number of complete decisions. Longer available sequences are retained.

Set `Config.TRAFFIC_SCALE_FACTOR` in `config.py` to a finite number of at least 1. Its default `None` uses the number of hosts. A factor `X` multiplies every pooled PCAP packet timestamp and the trace duration by `X`, preserving the packet contents and byte count. With nine hosts, five-second decisions, and a minimum of 20 decisions, a short split is extended to 12 source seconds per pair. Scaling those windows produces 108 seconds and 21 complete decisions. Synthetic traffic preserves each source window's byte volume, stretches that window to `X` seconds, and divides its sending rate by `X`. A value of 1 produces the original timing. Synthetic transfer timeouts cover these longer delivery windows.

Each generated manifest contains host IDs, MACs, IPv4 addresses, scaled duration, decision interval, decision count, seed, and the resolved `traffic_scale_factor`. The decision count is calculated from the final scaled duration. PCAP manifests identify per-pair captures and retain actual one-second `mbps_by_second` bins after stretching. Synthetic manifests record `window_seconds` for interpreting their traffic series and scheduled TCP jobs. Each synthetic window's `frame_bytes` is its volume over that whole window, and `average_mbps` includes its duration. Both formats produce throughput PNGs with elapsed seconds, total Mbps, and each direction. IPv4, IPv6, ARP, and transport checksums are adjusted during address rewriting. Network generation reads host metadata directly from the manifests.

`--skip-pcaps` generates synthetic datasets while reading capture statistics directly, skipping both packet spooling and per-pair PCAP output. `--skip-train` generates test datasets only and also applies to the congestion command. Random pools use the seed, split, and capture name, so the same test datasets are generated when either output option is selected.

```sh
model/.venv/bin/python -m data_preparation default --hosts 50 --skip-pcaps --skip-train
model/.venv/bin/python -m data_preparation congestion --hosts 50 --network networks/v4.json --skip-train
```

The test-only output contains DC, Megalodon, and the traffic-load variants. Congestion preparation adds the selected test variants. PCAP-based training and evaluation use the captures produced by a full preparation run.

| Dataset                                         | Split          | Meaning                              |
| ----------------------------------------------- | -------------- | ------------------------------------ |
| `dc_pcap.json`                                  | train and test | Remapped Digital Corpora PCAPs       |
| `dc_t0.json`                                    | train and test | Digital Corpora synthetic traffic    |
| `mg_pcap.json`                                  | test           | Remapped Megalodon PCAPs             |
| `mg_t0.json`                                    | test           | Megalodon synthetic traffic, also T2 |
| `dc_t1.json`                                    | test           | 0.5 times DC T0 traffic              |
| `dc_t3.json`                                    | test           | 1.5 times DC T0 traffic              |
| `dc_t4.json`                                    | test           | 2 times DC T0 traffic                |
| `dc_t0_trenv2.json` through `dc_t0_trenv5.json` | train and test | Regional congestion variants         |

Generate congestion variants after V0 has been partitioned.

```sh
model/.venv/bin/python -m data_preparation congestion --hosts 9 --network networks/v0.json
```

TrEnv 1 is DC T0. TrEnvs 2, 3, and 4 increase traffic originating in the first, second, and third controller domain by 1.5. TrEnv 5 increases traffic from all domains by 1.5. Domain order follows the network's controller list. Congestion preparation checks the host directions against the selected network.

Traffic-load and congestion multipliers apply to the already stretched synthetic data, retaining its window durations and global time scale.

## Network preparation

Generate V0 and V3 from the prepared source host metadata.

```sh
model/.venv/bin/python -m network_preparation v0 --dataset datasets/9_hosts/train/dc_t0.json
model/.venv/bin/python -m network_preparation v3 --dataset datasets/9_hosts/train/dc_t0.json
```

`network_preparation/source_v0.json` preserves the source's 13 switches, 16 inter-switch links, port map, capacities, domain assignments, and host attachment positions. Generation replaces its host identities and addresses with the dataset's nine hosts. Controller slots stay ordered c0, c1, c2, controlling s12, s11, s13. Controller locations minimize mean hop distance within the existing domains.

V3 retains that graph and its host attachments, adding three switches and six inter-switch links. The added switches connect to s1/s6, s3/s7, and s5/s9. New capacities use the configured seed and the 58 to 100 Mbps range. This yields 16 switches and 22 inter-switch links.

Generate V1 and V2 with the random graph command and assign their output names explicitly.

```sh
model/.venv/bin/python -m network_preparation random --nodes 13 --links 16 --dataset datasets/9_hosts/train/dc_t0.json --name v1
model/.venv/bin/python -m network_preparation random --nodes 13 --links 18 --dataset datasets/9_hosts/train/dc_t0.json --name v2
```

The random generator creates a connected simple graph with the requested counts. Topology Zoo processing collapses directed and parallel edges, removes self-loops and marked hosts, and retains the largest connected component. It prints switch and inter-switch link counts. To prepare V4 with 18 hosts, first prepare the corresponding datasets, then generate its network.

```sh
model/.venv/bin/python -m data_preparation default --hosts 18 --min-decisions 20
model/.venv/bin/python -m network_preparation zoo --topo GtsCe --dataset datasets/18_hosts/test/dc_t0.json --name v4
```

Random and Zoo preparation also accept `--hosts` to allocate conventional host addresses directly. Omitting `--topo` processes all bundled Zoo files. `--output-dir` and `--seed` control output placement and random choices.

`config.py` sets the number of domains, controllable switches per domain, and balanced Louvain parameters. Partitioning uses link capacities sampled uniformly in integer Mbps from 58 to 100. A final modularity pass merges connected communities to the requested count. The move-capacity constraint applies before that merge. Controller placement minimizes mean hop distance and orders slots by placement switch number. Controllable switches are chosen by descending degree with connectivity checked under every disable combination. Hosts fill peripheral access switches round-robin across domains, with one host per available access switch.

Each network writes a version-1 JSON, final SVG, and pre-merge SVG. The final plot distinguishes hosts, switches, controller locations, controllable switches, domains, and Mbps capacities. Generated source and V3 pre-merge diagrams show their preserved domains. The RL action remains one binary support-switch action per controller, using the first controllable-switch entry.

## Runtime contract

Start Ryu before Mininet and pass the identical network JSON to both launchers. Training and evaluation require only its filename, resolved under `networks/`.

```sh
ryu_controller/.venv/bin/python ryu_controller/launch_controllers.py --scenario networks/v0.json
sudo python3 mininet_nets/launch_topology.py --scenario networks/v0.json
```

The model reads control addresses from `.socket` and `mininet_nets/.socket`. Override them with `--ryu-socket` and `--mininet-socket`. The network fingerprint is checked before traffic starts. Controller IDs and order must match the model's encoder on every topology. Different switch and host counts fit the same model when the logical controller slots remain compatible.

Datasets are sent as job lists through the Mininet socket. Switching datasets leaves the topology fingerprint stable. Synthetic traffic uses one asynchronous receiver and one scheduled sender process per participating host. Comparison evaluations record individual flow measurements. PCAP replay preserves prepared packet timestamps with `tcpreplay --multiplier=1`, including any initial idle interval. Comparison evaluations capture transmitting and receiving interfaces for payload-delivery measurements. Training and embedding collection use controller telemetry and skip delivery captures. Actions update forwarding through the controllers.

Each decision applies all agent actions, waits for acknowledgments and a fresh telemetry window, then computes rewards. FeatherGraph embeds each domain into `4 * order * eval_points` features and concatenates them in controller order. Default settings produce 12 features for three agents. Training fits normalization on source training traffic and saves it in the checkpoint and inference model.

By default, an episode uses every complete decision interval available after warm-up. After its last decision, `drain_seconds` allows pending delivery under the final forwarding configuration. Completed deliveries contribute FCT measurements. Incomplete deliveries are ignored after draining.

## Training and resumption

```sh
model/.venv/bin/python -m model.train --network v0.json --traffic synthetic --output dc_t0
model/.venv/bin/python -m model.train --network v0.json --traffic pcaps --output dc_pcaps
model/.venv/bin/python -m model.train --network v0.json --traffic curriculum --output dc_full
```

`--traffic` chooses source training manifests automatically. Synthetic training uses `dc_t0`, PCAP training uses `dc_pcap`, and curriculum training advances through T0 and TrEnvs 2 to 5. Calibration uses all-on source training traffic from the selected conditions. Each run pretrains the reward estimator and freezes it while the actors and critics learn. Pretraining cycles through the selected conditions. The policy episode budget is split evenly across curriculum stages, with any remainder assigned to earlier stages. Curriculum requires at least five policy episodes.

Use `--episodes` and `--pretrain-episodes` for direct budget overrides. `--settings custom.json` merges a partial JSON file into the defaults. Unknown setting names are rejected. The full defaults live in `settings/default.json`.

| Setting                                  | Default                                                         |
| ---------------------------------------- | --------------------------------------------------------------- |
| Decision interval                        | 5 seconds                                                       |
| Warm-up                                  | 0 seconds                                                       |
| Drain                                    | 5 seconds                                                       |
| Calibration                              | 1 episode per selected training condition                       |
| Reward-estimator pretraining             | 5 episodes                                                      |
| Policy training                          | 30 episodes                                                     |
| Actor / critic / estimator learning rate | 0.0001 / 0.001 / 0.001                                          |
| Batch size                               | 8                                                               |
| Pruning                                  | dynamic, 5 initial dense rounds, then 30% to 70% over 20 rounds |
| Evaluation                               | 5 episodes per repetition, 10 repetitions                       |
| Seed                                     | 42, overridable with `--seed`                                   |

The reward uses delivered throughput divided by 8 Mbps, clipped to one, minus 0.5 times modeled power divided by 200 W, clipped to one. Modeled power adds 30 W per active switch, 1 W per active port, and 0.05 W per Mbps of switch egress. Configure these values under `reward`.

Consensus mixes peers equally, keeping actors local. Pruning affects outgoing copies of shared network weights. The first layer and biases remain dense. Static pruning removes 70% of eligible outgoing weights. Mode `none` sends dense contributions. Dynamic pruning counts communication rounds across pretraining and policy learning. The frozen-estimator phase shares critic parameters and running average reward.

```sh
model/.venv/bin/python -m model.train --network v0.json --traffic synthetic --output dc_t0_no_pruning --settings settings/no_pruning.json
model/.venv/bin/python -m model.train --network v0.json --traffic synthetic --output dc_t0_static_pruning --settings settings/static_pruning.json
```

Every decision prints the mean agent reward and completed/remaining decision counts. Episode completion prints cumulative reward, completed/remaining episodes, and ETA. Cumulative reward is the sum of mean agent rewards over that episode's joint decisions. Training CSVs contain `episode,cumulative_reward` for policy-learning episodes. Pretraining and calibration have separate run records.

The named inference file is saved at `models/<name>`. Its CSV and PNG are `models/<name>.csv` and `models/<name>.png`. The latest episode-boundary checkpoint is `models/checkpoints/<name>.pth`, with logs in `<name>.run/`. A per-model file lock prevents concurrent writers for the same name. Checkpoints include optimizer, action RNG, estimator, pruning round, schedule, input hashes, and reward history. Atomic writes keep checkpoint and CSV recovery consistent. Resume uses the same model name and original command arguments plus `--resume`.

```sh
model/.venv/bin/python -m model.train --network v0.json --traffic curriculum --output dc_full --resume
```

`--workers local` keeps learners in-process. The default `process` creates one spawned worker per agent. `--device auto` selects CUDA when available, with `cpu`, `cuda`, and `cuda:N` available explicitly. Loading checkpoints maps tensors through CPU and supports resuming on a different device.

## Six evaluations

All figures and evaluation measurements are written under `eval/`. Models are selected by inference filename from `models/`. Host counts and test dataset paths follow the chosen network. The comparison defaults are `dc_t0`, `dc_pcaps`, and `dc_full`, in that order. `--models` accepts three replacement names for those roles.

1. `model.eval_topology_embeddings` collects 16 post-reset observations from DC synthetic T0 with every switch on. Run it once for each manually started topology. It stores `eval/topology_embeddings_v<n>.json` and jointly projects the 80 observations with seeded t-SNE into `eval/topology_embeddings.png`.
2. `model.eval_traffic_embeddings` uses V0 and takes the first post-reset observation with delivered traffic from T0, T1, Megalodon synthetic T2, T3, T4, and TrEnvs 2 to 5. It saves one nine-point t-SNE figure and its raw embeddings and coordinates.
3. `model.eval_datasets` compares the three models with an all-on baseline on V0. It produces nine grouped bar charts, covering three metrics for each of the source, load, and congestion groups. Source charts have four groups, load charts have three groups, and congestion charts have four groups. Each group has three model bars.
4. `model.eval_topologies` evaluates DC PCAPs for the currently started topology. After V0 through V4 have been collected, it produces three metric charts with five topology groups. Each invocation also writes `dc_t0_v<n>.csv`, `dc_pcaps_v<n>.csv`, and `dc_full_v<n>.csv` with matching PNGs. Each CSV contains the mean raw episode return across repetitions. Five topologies produce 15 CSVs and 15 reward plots.
5. `model.eval_pruning` reads `models/dc_t0.csv`, `models/dc_t0_no_pruning.csv`, and `models/dc_t0_static_pruning.csv`. It writes `eval/pruning_reward.png` and `eval/pruning_relative.png`.
6. `model.eval_switches` runs one episode each for the three models and the all-on baseline on a provided network. `--steps` specifies the exact number of joint decisions. Each point is the number of enabled controllable switches after that decision, calculated as `sum(actions)`. The graph legend includes each method's final count. Outputs default to `eval/switches_<network>.png` and `.json`, with `--name` available to set another prefix. The JSON records counts at every step and the final counts, input hashes, seed, and settings. `--dataset` defaults to `dc_t0` and resolves a prepared test dataset using the network's host count. Each method uses the same seed and traffic schedule. Episodes longer than the dataset include idle time after replay finishes. Switch-count collection uses controller telemetry and skips delivery captures.

```sh
model/.venv/bin/python -m model.eval_topology_embeddings --network v0.json --model dc_t0
model/.venv/bin/python -m model.eval_traffic_embeddings --model dc_t0
model/.venv/bin/python -m model.eval_datasets --models dc_t0 dc_pcaps dc_full
model/.venv/bin/python -m model.eval_topologies --network v0.json --models dc_t0 dc_pcaps dc_full
model/.venv/bin/python -m model.eval_pruning
model/.venv/bin/python -m model.eval_switches --network v0.json --steps 20 --models dc_t0 dc_pcaps dc_full
```

For topology collection, stop both launchers, restart Ryu first and Mininet second with the next JSON, and invoke the two topology commands for that network. V4 automatically uses its larger-host-count test datasets. The final topology collection produces aggregate figures. Redraw saved results with `--plot-only` on either topology evaluation. `--name` selects the output prefix for evaluations 1 through 4.

The dataset and topology comparisons use identical prepared traffic for each condition. Repetition index `i`, starting at zero, uses `seed + i` for every method. Override `--episodes`, `--repetitions`, and `--seed` on those two comparisons. Policies remain frozen and use greedy actions. Bar error bars are 90% percentile bootstrap intervals across paired repetitions.

FCT averages completed payload-delivery times. PCAP completion tracks delivery of unique TCP payload ranges within the prepared replay. Mean power is weighted by measured decision duration. The metrics follow `FCT_gain = 1 - FCT_model / FCT_baseline`, `PS = 1 - power_model / power_baseline`, and `efficiency = (FCT_gain / max_mean_FCT_gain) / (1 - PS)`. The normalization maximum is taken across the three models within each dataset/topology condition. Power-saving bars show `100 * PS`. Metrics with no completed baseline flows or undefined denominators have empty numeric values.

Pruning comparison aligns common episode numbers. Relative reward is `100 * (reward_model - reward_none) / abs(reward_none)`. A no-pruning reward of zero leaves that relative value undefined. The no-pruning reference is zero percent wherever the denominator is defined.

## Verification

The unit suites use small fixtures and mocked runtime clients. They can be run without launching emulation or controller processes.

```sh
model/.venv/bin/python -m unittest discover -s model/tests
model/.venv/bin/python -m unittest discover -s data_preparation/tests
model/.venv/bin/python -m unittest discover -s network_preparation/tests
python3 -m unittest discover -s mininet_nets/tests
ryu_controller/.venv/bin/python -m unittest discover -s ryu_controller/tests
```

Network generation, full capture preparation, training, and evaluation are explicit notebook or command-line operations. Live forwarding, replay timing, telemetry, and completion measurements require a Mininet/Ryu integration run.
