# EAGLE replica documentation

EAGLE experiments with adaptive capacity in a software-defined network. Mininet runs a fixed set of Open vSwitch datapaths and links. Three Ryu controllers exchange topology information, collect switch counters, and program forwarding rules. A reinforcement-learning policy chooses whether each controller should use its supporting switch for forwarding. A separate graph partitioning benchmark works on offline topology data.

This guide covers the repository's Python application code and the configuration it consumes. It skips the internals of `ding/`.

## Repository map

| Path | Responsibility |
| --- | --- |
| [`mininet_nets/`](mininet_nets/) | Build the emulated network and provide the controller's graph and OpenFlow helpers. |
| [`ryu_controller/`](ryu_controller/) | Run the three controller processes, exchange topology, collect statistics, and expose the environment API. |
| [`environment/`](environment/) | Convert one controller's HTTP responses into an observation and reward for one agent. |
| [`model/`](model/) | Coordinate all three agents, embed their graph observations, configure learning, and run training or checkpoint evaluation. |
| [`partitioning/`](partitioning/) | Load benchmark topologies, run Balanced Louvain, and render communities to a PNG. |

## Setup and entrypoints

The project pins **Python 3.7** in [`pyproject.toml`](pyproject.toml), [`.python-version`](.python-version), and [`mise.toml`](mise.toml). With that interpreter available, sync the locked Python dependencies from the repository root.

```sh
uv sync --locked --no-install-project
```

Run the Python files directly from their component directories. The declared `eagle-replica` console script points to `eagle_replica:main`, but that module is absent in the checkout. The training and evaluation scripts prepend the **parent of the working directory** to the import path, so launching them from `model/` selects the repository's local dependencies.

For live emulation, provision Mininet and Open vSwitch on the host and make `ryu-manager` available to the controller launch shell. Ryu is excluded from the lockfile workflow. The original split dependency lists are [`original_reqs/ryu_requirements.txt`](original_reqs/ryu_requirements.txt), which lists Ryu 4.34, and [`original_reqs/rl_requirements.txt`](original_reqs/rl_requirements.txt). Mininet needs privileges to create its switches, links, and host namespaces.

1. Launch all three controllers from a shell where `ryu-manager` resolves. The working directory supplies `controller.py`, `config.ini`, and the controller's `topology.json`.

   ```sh
   (cd ryu_controller && ./start_controllers.sh)
   ```

2. In another terminal, start Mininet with a Python interpreter that can import Mininet. Keep its CLI open for the duration of the experiment.

   ```sh
   (cd mininet_nets && sudo python custom_topology_v1.py)
   ```

3. Wait for the datapaths to register, for the controllers to exchange their graphs, and for the controller statistics to become available. If the experiment calls for traffic, start it in the emulated hosts before collecting results. [`start_traffic.py`](mininet_nets/start_traffic.py) is a Mininet CLI snippet and expects the host objects to exist in the CLI context. It starts local shell scripts on `h1`, `h2`, `h4`, and `h7`. Those scripts and their traffic inputs are absent from the tracked checkout.

   ```text
   mininet> py exec(open('start_traffic.py').read())
   ```

4. Start training or checkpoint evaluation from `model/`. Both change the same emulated network and should be run separately.

   ```sh
   (cd model && ../.venv/bin/python train_decentralized.py)
   # Set model_path in model/test_decentralized.py before evaluation.
   (cd model && ../.venv/bin/python test_decentralized.py)
   ```

5. Exit the Mininet CLI to let the script stop the network. The controller launch script backgrounds all three processes. [`stop_controllers.sh`](ryu_controller/stop_controllers.sh) uses `sudo pkill -f controller.py`, so its scope includes any matching command line on the host.

## Mininet topology and switch ownership

[`custom_topology_v1.py`](mininet_nets/custom_topology_v1.py) creates the topology in Python using `OVSKernelSwitch` and bandwidth-limited `TCLink` links. It creates nine hosts (`h1`–`h9`), ten base switches (`s1`–`s10`), and three supporting switches (`s11`–`s13`). It configures static ARP entries and assigns each switch to its remote controller.

| Controller | OpenFlow | HTTP | Peer exchange | Assigned switches | Supporting path |
| --- | --- | --- | --- | --- | --- |
| `c0` | 6633 | 8080 | 8000 | `s1`, `s2`, `s6`, `s10`, `s12` | `s6 → s12 → s10` |
| `c1` | 6634 | 8081 | 8001 | `s5`, `s9`, `s11` | `s5 → s11 → s9` |
| `c2` | 6635 | 8082 | 8002 | `s3`, `s4`, `s7`, `s8`, `s13` | `s7 → s13 → s8` |

There are two node-link JSON files with the same basename. The launcher constructs Mininet directly in Python. `Configuration_V1` reads [`ryu_controller/topology.json`](ryu_controller/topology.json) when Ryu is launched from its directory. That copy includes the `bw` values used for utilization. [`mininet_nets/topology.json`](mininet_nets/topology.json) has a different set of link metadata.

When editing the topology, align switch names and datapath IDs, host MAC addresses, link endpoint order and port numbers, and bandwidth between the Python topology and the controller JSON. For links, `node1` and `node2` determine which endpoint uses `port1` and `port2`. Changing the order of Mininet link creation can affect those port assignments.

## Ryu controller and forwarding

[`controller.py`](ryu_controller/controller.py) defines two classes.

- **`SimpleMonitor13`** is the Ryu application. It registers connected datapaths, initializes their forwarding state, and runs two background loops. The monitoring loop requests port statistics from connected switches and flow statistics from switches with local hosts. The peer loop chooses its identity from the supporting datapath ID and exchanges node-link graphs and boundary edges over `nnpy` sockets. After receiving both peer graphs, it composes them with the local network, updates the known host MACs, and rebuilds forwarding.
- **`SimpleSwitchController`** is the WSGI adapter used by the agents. It exports a statistics snapshot and sends policy actions to the topology manager.

[`config_reader.py`](ryu_controller/config_reader.py) wraps `ConfigParser`. The shipped [`config.ini`](ryu_controller/config.ini) supplies `frequency_data = 1` for the polling loop. The other reader methods expect keys or sections that the current INI omits.

[`configuration_v1.py`](mininet_nets/configuration_v1.py) contains the controller's topology and routing logic.

- `topology` is the source graph. `net` represents the local active graph, and `global_net` incorporates peer topology. The separate `graph` used for observations is built with local nodes and boundary edges so link counters can be mapped to neighboring nodes.
- `act()` reads the first bit in an action. A change to the supporting switch's enabled state adds or removes it and its edges from the active graphs. The controller clears flow tables, installs forwarding again, and announces the topology change to peers. The Mininet datapaths and physical emulated links stay up.
- `route()` programs L2 flows for source and destination MAC pairs using shortest paths. Where several equal-length paths exist, it prefers paths that include the local supporting switch when available. Otherwise it uses a CRC32C hash of the host pair and MACs to select a path.
- `normalize_state()` maps port counters to node pairs and scales byte deltas by the link's configured bandwidth. It caps ratios at 1 and zeroes ratios below 0.01. The calculation uses the polling cadence implicitly and has no explicit elapsed-time divisor.

[`manageInterfaces.py`](mininet_nets/manageInterfaces.py) builds and sends OpenFlow flow-mod messages. The active routing path uses `Interfaces.add_flow()` and `Interfaces.remove_flows()`. The file also contains L2 and L3 interface-change helpers outside the main action path.

### HTTP interface

Each `DistributedEnv` addresses its controller at `127.0.0.1` on the HTTP port listed above.

| Method and path | Contract used by the environment |
| --- | --- |
| `GET /network/stats` | Returns JSON with `stats.data` (nested switch/neighbor utilization ratios), `throughput` (flow ID to byte delta), `demands` (space-separated source and destination host names to byte delta), and `topology` (NetworkX node-link data). |
| `PUT /model/actions` | Accepts a one-element JSON array, `[0]` to disable or `[1]` to enable the local supporting switch in the forwarding graph. A successful request echoes the action. |

The first statistics reply establishes a baseline. Later replies provide byte deltas. Flow counter decreases are treated as resets, and the new counter value is used as the delta. Startup can leave statistics unavailable until a complete sample is collected. The stats handler returns HTTP 400 for an empty `stats.data` and also assumes the snapshot has been initialized. The environment has no readiness retry loop. The `/network/list_switches` path is declared in code, but the WSGI adapter registers no handler for it.

## Reinforcement-learning pipeline

```text
train_decentralized.py or test_decentralized.py
  → NetworkEnv
    → ParallelEnv
      → DistributedEnv for each of 8080, 8081, 8082
        → Ryu HTTP API
          → OpenFlow datapaths in Mininet
```

### One agent: DistributedEnv

[`distributed_env.py`](environment/distributed_env.py) loads the matching `config_<port>.ini` relative to its own file. Each shipped config sets four interfaces, one supporting switch, an overprovisioning penalty of 5, and a four-second data interval. `HelpedSwitches` stores zero-based switch indices. The active action is sent to the controller selected by the port.

- **Observation.** `_get_state()` deserializes the node-link graph, attaches the reported utilization ratios as edge weights, and keeps the larger ratio if both directions update the same undirected edge. It caches nonzero flow deltas and returns the demands and weighted graph.
- **Action.** The discrete choices 0 and 1 map to `[0]` and `[1]`. `parallel_step()` sends the action immediately, and `parallel_info()` later fetches the state and computes the reward. The direct `step()` method combines both and sleeps for the configured data interval.
- **Reward.** The implementation takes the mean nonzero flow byte delta, multiplies by `8 / 1024 / 1024`, and divides by the normalization constant 40. It subtracts `0.01 × OPPenalty` for each enabled supporting switch, then scales by 100 and truncates to an integer. The byte delta is per sample, so changing the controller's polling interval changes the effective throughput scale.

### Three agents: ParallelEnv and NetworkEnv

[`parallel_env_net.py`](model/parallel_env_net.py) creates one `DistributedEnv` per controller on reset. During a step it sends all actions in order, waits a shared four seconds, then collects all observations, rewards, and demands. The four-second sleep here is hardcoded and separate from `FrequencyData`. Results are keyed by HTTP port strings, while inputs to `ParallelEnv.step()` are keyed by integer agent index. Carbon and path-flow utilities in this module are outside the active step path and expect additional carbon data.

[`multi_env.py`](model/multi_env.py) registers `NetworkEnv` as `multi_network`. It accepts a list of per-agent actions, converts them to integer actions, and returns a `BaseEnvTimestep` containing the new observation, a list of agent rewards, a shared done flag, and info. It accumulates the mean agent reward per step into the episode return and emits it when the configured step limit is reached.

For each controller graph, `NetworkEnv` relabels nodes as integers and runs `FeatherGraph(order=1, eval_points=1)`. It rounds and scales the embeddings, concatenates four features per controller in port order, and applies fixed per-feature min/max normalization to produce the 12-element `global_obs`. The fixed port list and normalization vectors make the three-controller setup part of the model contract. Changing only the agent count in the config is insufficient.

### Configuration, training, and checkpoint evaluation

- [`net_config.py`](model/net_config.py) exports `main_config` and `create_config`. Defaults include three agents, two actions per agent, 20 steps per episode, one collector and one evaluator environment, CPU execution, and a replay buffer of 20. The configured training experiment name is `decentralized_tr`.
- [`train_decentralized.py`](model/train_decentralized.py) compiles the config, initializes DI-engine, and wires an evaluator, step collector, replay-buffer pusher, off-policy learner, logger, and checkpoint saver into a synchronous task. The collector and evaluator manager factories both reference the same `NetworkEnv` instance and therefore share the same emulation. The saver is configured with the experiment name. The script contains no step that stages checkpoints under `saved_models/`.
- [`test_decentralized.py`](model/test_decentralized.py) is the live checkpoint evaluator. Set `model_path` to a trusted checkpoint containing a `model` entry. The shipped path contains a placeholder and the checkout has no tracked weights. The script loads the same path three times on CPU and passes the three model states to the model's loader. It runs two outer loops of 40 steps, resets the environment only once before both loops, and does not use `done` to break or reset. Its loop labels therefore differ from the environment's 20-step episode boundaries. This evaluator creates the same `NetworkEnv` as training, whose `ParallelEnv` constructs its agents with `training=True`.

## Standalone graph partitioning

This benchmark uses the included Topology Zoo GraphML files. It runs independently of Mininet, Ryu, and model checkpoints.

```sh
(cd partitioning && MPLBACKEND=Agg ../.venv/bin/python partitioning.py --topo GtsCe)
```

The command writes `partitioning/GtsCe.graphml_subnets.png`. Choose a registered basename without an extension. Run from `partitioning/` so the relative `topology-zoo/` inputs resolve.

| Python file | Behavior |
| --- | --- |
| [`benchmark_consts.py`](partitioning/benchmark_consts.py) | Registers topology names and parses `--topo`, `--capacity`, `--difference`, and `--rho`. The shipped registry contains GraphML entries and defaults to `GtsCe`. |
| [`problem.py`](partitioning/problem.py) | Loads graphs. The GraphML loader keeps the largest strongly connected component after converting to a directed graph, relabels nodes as integers, converts back to an undirected graph, and sets `capacity=1000.0` and `traffic=0` on edges. The file also has node-link JSON and edge-list readers. |
| [`balanced_louvain.py`](partitioning/balanced_louvain.py) | Implements `bpartitions()`, a generator that starts from singletons, evaluates node moves using weighted modularity and a balance term, and aggregates communities into a coarser graph for subsequent levels. Each yielded partition contains sets of original node IDs. |
| [`partitioning.py`](partitioning/partitioning.py) | Loads the selected graph, consumes the partition generator, builds subgraphs from the final partition, and hands the partition to the plotter. It passes fixed values `resolution=0.1`, `gamma=60`, `epsilon=10`, `rho=0.9`, and `seed=42`. The parsed capacity, difference, and penalty flags currently have no effect on the run. |
| [`vis.py`](partitioning/vis.py) | Colors the communities, draws the graph with a spring layout, and saves the PNG in the working directory. |

To benchmark another GraphML topology, add its file under `partitioning/topology-zoo/` and register its filename in `GRAPHS_NAMES`. The CLI resolves registered names rather than arbitrary paths.

## Focused verification

There is no tracked application test suite outside the excluded library. `model/test_decentralized.py` is an integration entrypoint requiring a live emulation and a checkpoint.

For a syntax check of the Python application files with the project's interpreter, run from the root.

```sh
.venv/bin/python -m compileall -q environment mininet_nets model partitioning ryu_controller
```

Use the standalone partitioning command above to check graph loading, community detection, and rendering together. For live changes, verify the controller statistics before launching a learner, then check that switching the supporting path changes the active forwarding graph while observations still resolve the expected port-to-node mappings.
