# Network preparation

Run from the repository root with the RL Python environment.

```sh
model/.venv/bin/python -m network_preparation.prepare random --nodes 13 --links 24 --hosts 9
model/.venv/bin/python -m network_preparation.prepare zoo --hosts 9
model/.venv/bin/python -m network_preparation.prepare zoo --topo GtsCe --hosts 9
```

The random command builds a connected switch graph with the specified switch and inter-switch link counts. The Zoo command prints the switch and link counts after collapsing directed and parallel links, removing self-loops and marked hosts, and keeping the largest connected component. GraphML switch IDs are recorded as `source_node` in the scenario.

Set the seed, domain count, controllable switches per domain, and Louvain move parameters in [config.py](../config.py). All link capacities are sampled uniformly from 58 to 100 Mbps in integer steps. Balanced Louvain uses these capacities as weights. A final modularity pass merges connected communities to the configured count when needed. The Louvain move-capacity setting applies before this final merge. Controller locations minimize mean hop distance to switches in their domain, and controller slots are ordered by placement switch number.

Controllable switches are chosen in descending order of switch degree, subject to each domain's quota and connectivity under every combination of disabled switches. Hosts are assigned round-robin across domains to the lowest-degree available access switches. Each access switch can take one host, and controllable switches have no host attachments.

Each run writes a version-1 scenario JSON, a final network SVG, and a `-pre-merge.svg` to `networks/`. Both diagrams use the same node positions. The pre-merge diagram labels the Louvain communities that enter the final merge. Use `--output-dir` for a different directory. Pass the JSON as `--scenario` to the launchers, the RL runner, and base data preparation, or as `--network` to regional data preparation. The generated scenario includes an idle traffic profile. See [data preparation](../data_preparation/README.md) for attaching traffic profiles.

Each controller declares its location in `placement_switch` and an ordered `controllable_switches` list. The current RL and emulation runtimes use the first controllable switch as `support_switch` and expose one binary action per controller.
