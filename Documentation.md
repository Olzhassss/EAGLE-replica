# EAGLE implementation guide

## Runtime boundaries

Mininet runs with the host's `sudo python3`. Ryu runs with Python 3.7 in `ryu_controller/.venv/`. Learning, evaluation, analysis and partitioning use modern Python in `model/.venv/`. Each project owns its Python code and dependencies. Shared data travels through the versioned scenario file, OpenFlow, and HTTP. The local process supervisors use Unix sockets for experiment operations.

Follow the [root startup sequence](README.md#start-an-experiment) to start a trial. The three project guides cover prerequisites, commands, inputs and shutdown. [Mininet](mininet_nets/README.md) · [Ryu](ryu_controller/README.md) · [RL](model/README.md)

## Closed-loop experiment

1. The [Mininet launcher](mininet_nets/launch_topology.py) builds the same node and endpoint port layout that the [controller topology loader](ryu_controller/topology_state.py) reads. Mininet uses traffic-controlled links and static host ARP. Every switch can connect to all controllers. Configured ownership sets the primary for that domain.
2. [Ryu](ryu_controller/controller.py) exchanges versioned peer views and elects the physical owner of each logical slot. It routes on the graph of currently connected, enabled resources and up links. Logical slots stay in scenario order even when a peer takes over a failed controller's domain.
3. [Routing](ryu_controller/routing.py) chooses a single stable end-to-end path per ordered MAC pair. [OpenFlow adapters](ryu_controller/openflow_rules.py) reconcile only the changed EAGLE flow entries. Dispatch time and barrier time are distinct.
4. [Telemetry](ryu_controller/telemetry.py) groups multipart replies by poll round, uses two complete rounds of the same topology generation for counter deltas, and exposes the sample window. Delivered bytes come from destination attachment switches. Observations use directional utilization mapped back to weighted subnet edges.
5. The [joint environment](model/joint_environment.py) sends all slot actions, waits for their acknowledgments and the decision window, then reads snapshots that start after the action and contain the acknowledged revisions. It gathers fresh state per slot or fails the trial with the recorded error. For optional interface or OVS experiments it also coordinates physical actions through Mininet.
6. The [encoder](model/observations.py) embeds each subnet and concatenates the embeddings in logical slot order. The [reward module](model/reward.py) calculates the normalized throughput and modeled power objective. It records the components independently of evaluation metrics.

The [V0 scenario](scenarios/v0.json) has 13 switches, nine hosts, and three support paths. Slot c0 at HTTP 8080 manages s12, c1 at 8081 manages s11, and c2 at 8082 manages s13. The default launchers use this order for observations, actions, policy weights, and checkpoints.

## Learning and checkpoints

The [training entrypoint](model/train.py) uses plain PyTorch to make the learning and consensus steps visible. A worker owns one local actor, critic, reward estimator, optimizers, running reward estimate, and action RNG. Workers can run in one process or in separate local processes. The trainer relays peer contributions over the process transport. It logs serialized payload sizes for consensus separately from total local worker traffic.

The [learning update](model/learner.py) uses local reward in the critic target, the joint action and state in the estimator, and the estimated shared reward in the actor update. Curriculum stages choose whether to pretrain, freeze, or update the reward estimator concurrently. [Consensus](model/consensus.py) mixes the configured active peers' parameters and scalar mean reward, leaving actors local. Dense, static, and dynamic modes are available. Pruning packs the retained weights and a bit mask without mutating the tensor being sent. The first layer and biases remain dense.

The [checkpoint loader](model/checkpoints.py) restores the saved encoder and agent networks. Evaluation reuses the stored preprocessing and loads through standard PyTorch state dictionaries. Each target topology must use the same logical slot order as the source checkpoint.

## Data and evaluation

A [trial](model/trials.py) restores the intended initial state, waits for fresh observations, starts configured traffic, runs the policy, and stops managed traffic. The default synthetic TCP job gives a short reproducible data path without original research captures. [PCAP preparation](scenarios/README.md#run-the-digital-corpora-capture) converts the Digital Corpora gzip into per-host replay windows and records the source frame intervals, addresses, rate, and checksums in the generated scenario. PCAP jobs collect sender and receiver captures. Delivery analysis uses unique TCP payload ranges and keeps incomplete flows in the output.

The [sweep entrypoint](model/sweep.py) repeats named method and condition cells, and the [report entrypoint](model/report.py) computes matched-baseline FCT gain, power saving, and relative efficiency. Power coefficients are scenario inputs for study settings. Reports retain individual run outcomes and bootstrap intervals. Plotting reads the saved events and does not replay traffic.

[Offline partitioning](partitioning/README.md) exports community memberships and minimum-hop placements. It can generate a shared scenario from an existing template if every inferred domain has one configured supporting switch.

The [study input inventory](scenarios/README.md#published-protocol) identifies the missing original traces, target scenarios, comparison adapters and testbed details. The compatibility baseline is available at Git revision `2577b26`. The application no longer imports vendored DI-engine, and the tree is retained for the planned parity check.

## Verification commands

The contract tests can be run separately when verification begins. From the repository root:

```sh
python3 -m unittest discover -s mininet_nets/tests
ryu_controller/.venv/bin/python -m unittest discover -s ryu_controller/tests
model/.venv/bin/python -m unittest discover -s model/tests
model/.venv/bin/python -m unittest discover -s partitioning/tests
```

The live check uses the [root startup sequence](README.md#start-an-experiment) followed by a fixed-action trial on each support path, a bounded training run, and separate checkpoint evaluation. For DI-engine parity, capture the same graph, actions and learner batch at revision `2577b26`, then compare the serialized results before removing the vendored reference.
