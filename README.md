# EAGLE replica

This repository implements the Mininet control loop and multi-agent training protocol from Monaco et al., _Adaptive SDN Autoscaling via Generalizable Multi-Agent Reinforcement Learning With EAGLE_ (IEEE TNSM 2026, DOI `10.1109/TNSM.2026.3717361`).

| Project                             | Runtime                                       | Guide                                   |
| ----------------------------------- | --------------------------------------------- | --------------------------------------- |
| Mininet emulation and traffic       | System Python 3 with Mininet and Open vSwitch | [Mininet guide](mininet_nets/README.md) |
| Ryu OpenFlow controllers            | Python 3.7 in `ryu_controller/.venv/`         | [Ryu guide](ryu_controller/README.md)   |
| EAGLE learning and experiment tools | Modern Python in `model/.venv/`               | [RL guide](model/README.md)             |

## Start an experiment

Run these commands from the repository root in separate terminals after setting up the two virtual environments.

1. Start the three controller processes. Keep the launcher running.

   ```sh
   ryu_controller/.venv/bin/python ryu_controller/launch_controllers.py --scenario scenarios/v0.json
   ```

2. Start the emulated network with system Python. The launcher writes its control socket address to `mininet_nets/.socket`. Add `--cli` for an interactive Mininet prompt.

   ```sh
   sudo python3 mininet_nets/launch_topology.py --scenario scenarios/v0.json
   ```

3. Run a fixed policy with the deterministic synthetic flow.

   ```sh
   model/.venv/bin/python -m model.evaluate --mininet-socket "$(cat mininet_nets/.socket)" \
     --traffic synthetic --fixed 1,1,1 --episode-steps 8 --output runs/fixed-v0
   ```

   Use `model.train` for training and `model.evaluate --checkpoint PATH` for frozen evaluation. The RL runner starts and stops its traffic per trial. Use Ctrl-C in the Mininet and Ryu launcher terminals to stop their services.

[Documentation.md](Documentation.md) describes the project boundaries, control loop, and replication workflow. [Scenarios](scenarios/README.md) lists the shared data and the remaining research inputs. [Partitioning](partitioning/README.md) runs offline in the RL environment.

## Traffic trace execution

Prepare the local Digital Corpora capture and generate a scenario with separate training and held-out traffic profiles.

```sh
model/.venv/bin/python -m model.prepare_traces 5gb-tcp-connection.pcap.gz --output runs/corpora-v0
```

Start both launchers with `--scenario runs/corpora-v0/scenario.json`. Pass the same scenario to training or evaluation and select `--traffic pcap-train` or `--traffic pcap-eval`. See the [trace run commands and study inputs](scenarios/README.md#run-the-digital-corpora-capture).

For the study's Megalodon T2 evaluation, prepare `MegalodonChallenge.7z` with the [Megalodon recipe](scenarios/README.md#run-megalodon-for-t2) and reuse the Digital Corpora-trained checkpoint.

The current application uses PyTorch directly. The original DI-engine tree remains available for the planned compatibility comparison.
