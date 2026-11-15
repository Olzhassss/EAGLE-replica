# Scenarios and study inputs

[`v0.json`](v0.json) reconstructs the checkout's source network. Its controller list defines logical action and feature order. Launch each project with the same scenario file. Peers and the RL client use its SHA-256 to detect mismatched definitions.

## Scenario schema

| Field | Contract |
| --- | --- |
| `schema_version`, `id`, `provenance` | Version 1, a stable scenario name, and the source or derivation. |
| `controllers` | Ordered entries with `id`, `host`, `openflow_port`, `http_port`, `support_switch`, and a binary `initial_action`. Optional `listen_host` overrides the controller's local bind address. |
| `switches` | ID, integer datapath ID, and nominal controller ID. The Mininet launcher gives every switch access to all configured controllers for takeover. |
| `hosts` | ID, IP address with prefix length, and unique MAC. Each host attaches to one switch. |
| `links` | `source`, `target`, `source_port`, `target_port`, and `capacity_bps`. Use positive switch ports and port 0 for a single Mininet host interface. Optional `partition_weight` controls the partitioner. |
| `telemetry` | `poll_seconds`, `peer_poll_seconds`, and `peer_timeout_seconds`. The decision step is configured separately in the run settings. |
| `traffic` | Named lists of scheduled traffic jobs. Every job carries `id`, `kind`, `source`, `target`, `start_seconds`, and `rate_bps`. |

A synthetic `tcp` job adds `size_bytes` and can set `timeout_seconds`. A `pcap` job adds a `path` relative to the scenario JSON and the prepared file's `sha256`. Record the original capture ID and hash, time offsets of the cut segment, and address rewrite in that job's provenance. Prepare one-way TCP segments and rewrite Ethernet addresses to the configured hosts. Replay data should use one connection per four-tuple and avoid sequence space reuse inside a job.

## Run the Digital Corpora capture

From the repository root, prepare the local `5gb-tcp-connection.pcap.gz` using the RL environment. The command reads gzip directly. It exports one-way TCP payload packets, rewrites the Ethernet addresses for the scenario hosts, and writes replay files and job provenance next to the generated scenario.

```sh
model/.venv/bin/python -m model.prepare_traces 5gb-tcp-connection.pcap.gz --output runs/corpora-v0
```

The default [recipe](corpora-example.json) selects the bulk direction from `69.163.156.173:80` to `8.30.79.4:57244`. It uses original frames 1–20,000 for `pcap-train` and 20,001–40,000 for `pcap-eval`. Both profiles map traffic to `h1 → h4` at 8 Mbps. These are checkout examples using separate windows of the same TCP connection. The output directory must be new.

Start the controllers and Mininet in separate terminals. The Mininet host needs `tcpreplay` and `tcpdump`.

```sh
ryu_controller/.venv/bin/python ryu_controller/launch_controllers.py \
  --scenario runs/corpora-v0/scenario.json
```

```sh
sudo python3 mininet_nets/launch_topology.py --scenario runs/corpora-v0/scenario.json
```

In a third terminal, set `MININET_SOCKET` to the path printed by Mininet and run the compatibility-profile example.

```sh
model/.venv/bin/python -m model.train --scenario runs/corpora-v0/scenario.json \
  --mininet-socket "$MININET_SOCKET" --traffic pcap-train \
  --episodes 3 --episode-steps 8 --output runs/pcap-train

model/.venv/bin/python -m model.evaluate --scenario runs/corpora-v0/scenario.json \
  --mininet-socket "$MININET_SOCKET" --traffic pcap-eval \
  --checkpoint runs/pcap-train/checkpoint.pth --episode-steps 8 \
  --keep-captures --output runs/pcap-eval
```

Replay restarts each episode. Size the episode to cover the latest scheduled start and `provenance.estimated_replay_seconds`, with time for a final observation. The example uses roughly 32 seconds at four seconds per decision. Flow records in `events.jsonl` report offered and received payload bytes and the replay completion time. `--keep-captures` retains sender and receiver PCAPs. The IP and TCP headers stay as captured because routing and replay use the rewritten MAC addresses. Completion measures delivery of the selected payload window at the capture interface.

### Configure study traffic

Copy the example recipe and pass it with `--recipe` to prepare other windows and host pairs. Each job uses inclusive, 1-based frame numbers from the input capture, a host pair, a start offset in seconds, and a rate in bps. Set `flow` to the captured IPv4 TCP direction. Use separate jobs for different ordered host pairs to compose the offered load in a profile. Split large windows before their TCP sequence space repeats. The generated jobs record original and rewritten SHA-256 hashes, source frame ranges, payload counts, and estimated replay duration. Extra captures can be prepared in a second invocation by supplying the first generated scenario with `--scenario` and a recipe with new profile names.

For the published protocol, prepare T0 from Digital Corpora and map its five training conditions to the `learning.stages` in a completed copy of [study.json](study.json). Use TrEnv 1 for uncongested traffic, TrEnvs 2–4 for the individual subnet hotspots, and TrEnv 5 for concurrent congestion in all three subnets. Fit normalization on source-only calibration runs, then train with `--settings` pointing to that completed profile and `--normalize-from` pointing to the calibration events. Reuse the source checkpoint with `model.evaluate` for the held-out scenarios. Supply the lower-utilization Corpora window for T1, Megalodon for T2, and higher-utilization Corpora windows for T3 and T4. Run the comparisons with matched traffic, fixed-action baselines and ten repetitions per condition.

The paper gives a five-second decision interval and the curriculum structure. Exact frame intervals, host mappings, replay rates, phase lengths, reward normalization, and initial baseline settings still require provenance. See [Published protocol](#published-protocol) for the remaining inputs.

### Manually prepared captures

Digital Corpora publishes packet captures in its [corpus browser](https://downloads.digitalcorpora.org/corpora/), under `packets/`. The paper and original README name a 5 GB TCP trace without giving its filename, checksum, or exact intervals. Record the capture and intervals you actually use. The `nanomsg-1.3.0.tar.gz` file is the [nanomsg library source release](https://github.com/nanomsg/nanomsg/releases) and contains no PCAP.

1. Start with a small Ethernet capture containing TCP payload. With `tshark`, select a complete stream and one direction. Set `STREAM`, `SRC_IP`, and `DST_IP` from the original capture before running the following example. It maps that direction to V0 hosts `h1` and `h4`.

   ```sh
   mkdir -p traces
   tshark -r original.pcap -Y "tcp.stream == $STREAM && ip.src == $SRC_IP && ip.dst == $DST_IP" \
     -F pcap -w traces/source-direction.pcap
   tcprewrite --infile=traces/source-direction.pcap --outfile=traces/train-h1-h4.pcap \
     --enet-smac=f6:c0:28:8f:a5:9a --enet-dmac=f6:c3:e5:59:2a:6d
   sha256sum traces/train-h1-h4.pcap
   ```

2. Copy `scenarios/v0.json` to `scenarios/v0-pcap.json`. Add an entry under its existing `traffic` object. Replace the hash with the output for the rewritten PCAP. The example rate of 20 Mbps is a starting point for exercising replay. Use a disjoint stream or time window to build a `pcap-eval` profile the same way.

   ```json
   "pcap-train": [
     {
       "id": "train-h1-h4",
       "kind": "pcap",
       "source": "h1",
       "target": "h4",
       "path": "../traces/train-h1-h4.pcap",
       "sha256": "REPLACE_WITH_PREPARED_PCAP_SHA256",
       "start_seconds": 1,
       "rate_bps": 20000000
     }
   ]
   ```

3. Start both launchers with `--scenario scenarios/v0-pcap.json` using the [root startup sequence](../README.md#start-an-experiment). The Mininet host needs `tcpreplay` and `tcpdump`. Set `MININET_SOCKET` to the path printed by Mininet, then select the trial profile. Set `--episode-steps` to cover the replay start and duration plus time for a final observation. The compatibility profile schedules four seconds per decision and replay restarts each episode.

   ```sh
   model/.venv/bin/python -m model.train --scenario scenarios/v0-pcap.json \
     --mininet-socket "$MININET_SOCKET" --traffic pcap-train --output runs/pcap-train
   model/.venv/bin/python -m model.evaluate --scenario scenarios/v0-pcap.json \
     --mininet-socket "$MININET_SOCKET" --traffic pcap-eval \
     --checkpoint runs/pcap-train/checkpoint.pth --keep-captures --output runs/pcap-eval
   ```

The capture supplies live traffic in Mininet. EAGLE learns from the resulting controller observations, actions, and rewards. The evaluator computes replay completion from the sender and receiver captures. Use `events.jsonl` for flow records and `--keep-captures` to retain the raw packet evidence.

## Run settings

[`compatibility.json`](compatibility.json) preserves the checkout's four-second decision interval, legacy observation preprocessing, and truncated reward with an enabled-switch penalty. The corrected controller and learner have their own implementations and require a future parity comparison.

[`study.json`](study.json) specifies the reported five-second decision interval, Feather dimensions, actor and critic learning rates, reward coefficient, dynamic pruning endpoints, and five curriculum stages after estimator pretraining. Copy it and fill the null fields using recovered experiment values. Source calibration uses `model.evaluate --identity-normalization` followed by `model.train --normalize-from`.

Each training stage can specify `active_agents` as a subset of logical slots participating in learner updates. The trainer still produces an action for every slot. Optionally supply `learning.mixing_weights` as one map of peer ID to weight per receiving slot. Each row is renormalized over active contributors. The default mixes all active agents equally.

Each settings profile can specify a resource mode (`forwarding`, `interfaces`, or `ovs`), and a list of failure events. Events run before their zero-based step `step`. Specify `kind: "link"`, `source`, `target` and boolean `up` for a link event. Use `kind: "controller"`, `controller_id` and boolean `up` for a controller outage or restore. For modeled carbon, specify `carbon.intensity_grams_per_kwh` for every active link endpoint and switch in a power-enabled configuration.

## Published protocol

| Study evidence | Existing inputs and remaining work |
| --- | --- |
| V0 and source run (Table III, Figs. 5–6) | The checked-in graph matches the reported 13-switch, 16-link counts. The preparation tool accepts the local Digital Corpora capture. Confirm original assignments, initial conditions, trace windows, host mappings, five training profiles, and baseline settings. |
| Target topologies V1–V4 (Figs. 4a, 7, 9) | Supply the authors' V1–V3 generators or topology assets, seeds and controller assignments. Bundled GtsCe supports graph partitioning. Its runnable V4 scenario still needs the study's hosts, capacity and controller mapping. |
| Target traffic T1–T4 (Figs. 4b, 8) | Supply the exact Corpora and Megalodon segments and replay settings. Keep target captures out of source calibration and training. |
| Comparison methods | Supply the published static configuration and adapters or measured data for DeepConf, SmartFCT, Mystique, CollaQ and CATE. Use the same condition name, windows and seeds for matched comparisons. |
| Learning and power | Recover the estimator data and stopping schedule, normalization bounds, power coefficients, carbon intensity inputs and any original checkpoints. Mark any reconstructed settings in run provenance. |
| Failures and latency (Figs. 10, 12) | Use explicit link and controller events and the selected physical resource mode. Supply the original failure schedule. The published latency runs use 25 repetitions. |
| Embedding and pruning sensitivity (Figs. 13–14) | Record each order, sample-point count and pruning mode in a separate settings profile. Train a checkpoint of matching width. Use the same traffic and seed schedule across settings. |
| NSFNET / FABRIC (Fig. 11) | Supply the testbed mapping, replay and capture data, and reachable controller endpoints. The evaluator accepts external traffic and flow records with a frozen checkpoint. |

Main comparisons use 10 runs per condition and method. Keep raw artifacts and label replication results by the provenance of their inputs.
