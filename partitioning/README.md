# Balanced Louvain and controller placement

Use the RL project's modern Python environment. Run from the repository root:

```sh
MPLBACKEND=Agg model/.venv/bin/python -m partitioning --topo GtsCe
```

The default CLI prints placement JSON and writes `GtsCe.graphml_subnets.png`. Use `--output FILE` to save the placement and `--no-plot` to skip rendering. Use `--input FILE` for a GraphML file or a version-1 scenario JSON. For a scenario, host attachments are omitted during partitioning. GraphML is projected to a simple undirected graph with parallel weights summed, and the largest connected component is retained.

`balanced_louvain.py` moves nodes or contracted node sets by modularity gain, subject to the configured capacity and imbalance penalty. `placement.py` chooses a switch within each community that minimizes mean hop distance to that community's switches. The CLI defaults of capacity 60, difference 10, rho 0.9, resolution 0.1 and seed 42 come from the original standalone benchmark. The study reports capacity 6, difference 2 and rho 0.01 for its source network. Its resolution and seeds require corresponding provenance.

```sh
model/.venv/bin/python -m partitioning --input scenarios/v0.json \
  --capacity 6 --difference 2 --rho 0.01 --no-plot --output placement.json
```

Pass `--scenario-template scenarios/v0.json --export-scenario PATH` to write controller assignments back into a scenario. A template supplies IDs, controller endpoints, supporting switches, hosts, ports, bandwidth, and traffic. The export requires one configured supporting switch per inferred community. Use a scenario with the published assignments to evaluate the study checkpoint.

The GtsCe fixture represents 149 switches and 193 undirected edges. Bidirectional representation gives 386 arcs, the count reported for V4 in the study.
