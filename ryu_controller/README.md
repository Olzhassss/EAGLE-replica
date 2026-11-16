# Ryu OpenFlow project

Create a dedicated Python 3.7 virtual environment. From the repository root:

```sh
python3.7 -m venv ryu_controller/.venv
ryu_controller/.venv/bin/python -m pip install 'setuptools==57.5.0' wheel
ryu_controller/.venv/bin/python -m pip install --no-build-isolation -r ryu_controller/requirements.txt
ryu_controller/.venv/bin/python ryu_controller/launch_controllers.py --scenario scenarios/v0.json
```

The launcher uses its own Python 3.7 interpreter for each Ryu child and manages only those children. Ctrl-C shuts them down. For separately placed controllers, run the launcher with `--controller ID` on each host and use a scenario with reachable endpoints and local bind addresses. Synchronize testbed clocks for sample-window and action-revision comparisons. The default V0 endpoints bind to loopback for a single-machine lab.

## Control and measurements

- `controller.py` registers the OpenFlow event handlers and HTTP interface. It checks datapath ports against the scenario, reconciles topology across peers, establishes an OpenFlow master for each owned switch, and waits for a barrier before sampling new routes.
- `topology_state.py` separates configured, active, and agent observation graphs. A logical slot survives controller loss because another live controller can take responsibility for its switches. Topology presence frames use a scenario fingerprint, sender identity, incarnation, and sequence. Startup expects all configured controllers and switches to appear before enabling the loop. Failover applies after startup and peer expiry.
- `routing.py` plans a deterministic end-to-end shortest path for each ordered host MAC pair. It prefers an equal-cost path through the source domain's support switch, then chooses among remaining candidates with CRC32C. `openflow_rules.py` sends only changed EAGLE-owned rules.
- `telemetry.py` pairs full multipart replies by round. Counter resets and topology generations restart baselines where needed. Delivered flow bytes are attributed to the destination attachment switch so study throughput counts a flow once.

| Interface                                         | Purpose                                                                                                                                                                                                   |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /network/state`                              | Publish the controller's view of ports, slot owners, and action revisions.                                                                                                                                |
| `GET /network/stats?controller_id=c0&view=active` | Return a fresh snapshot with its window, topology, resource state, counters, and dispatch timings. The `configured` view supports legacy preprocessing. A pending sample or ownership change returns 503. |
| `PUT /model/actions?controller_id=c0&details=1`   | Accept `[0]` or `[1]` and return an acknowledgment with revision and accepted time. Without `details=1` the response echoes the array.                                                                    |

The supervisor writes a Unix socket for configured controller outages and restores into `.socket`. Supply it to the experiment runner as `--ryu-socket` when using controller fault events. After Mininet starts, the intended smoke check is to fetch statistics for each slot and toggle its supporting path from the [fixed-policy runner](../model/README.md).
