# Ryu controllers

The supervisor runs one Ryu process per configured controller. Ryu needs its
own Python 3.7 environment, separate from EAGLE and system Mininet. Create it
from the repository root:

```sh
python3.7 -m venv ryu/.venv
ryu/.venv/bin/python -m pip install 'setuptools==57.5.0' wheel
ryu/.venv/bin/python -m pip install --no-build-isolation -r ryu/requirements.txt
ryu/.venv/bin/python -m ryu --network networks/v0.json
```

Start Ryu before [Mininet](../mininet/README.md) using identical network
JSON. `--controller ID` selects controllers for a deployment across hosts; it
may be repeated. Configure reachable endpoints and synchronize host clocks for
telemetry windows and action revisions. Ctrl-C stops managed child processes.

| File | Responsibility |
| --- | --- |
| `supervisor.py` | Start, restore, and stop controller processes through a local socket |
| `eagle_controller.py` | Handle OpenFlow events, peer presence, master roles, reconciliation, and snapshots |
| `api.py` | Serve HTTP network state, telemetry, and action endpoints |
| `network.py` | Load the configured graph, host addresses, and port index |
| `topology_state.py` | Track action revisions, elect logical slot owners, and build active/local graphs |
| `routing.py` | Select stable shortest host paths, preferring enabled source-domain support switches |
| `openflow_rules.py` | Reconcile EAGLE-owned rules and request switch roles |
| `telemetry.py` | Pair complete polls, handle counter resets, and calculate delivered bytes/utilization |
| `tests/` | Topology, routing, action, counter, and OpenFlow serialization contracts |

| HTTP interface | Result |
| --- | --- |
| `GET /network/state` | Peer presence, slot ownership, ports, and action revisions |
| `GET /network/stats?controller_id=c0` | Coherent slot telemetry; HTTP 503 while unavailable |
| `PUT /model/actions?controller_id=c0&network=SHA256` | Apply `[mask]` and acknowledge its action revision and acceptance time |

Each logical slot controls the ordered `support_switches` list in its network
definition (schema version 2). Bit `i` of its integer mask enables list element
`i`: with two supports, masks `0`, `1`, `2`, and `3` select neither, the first,
the second, or both. Zero-support domains accept only `0`. Out-of-range masks,
booleans, and fractional values are rejected. Peer reconciliation and failover
preserve the full mask and its action revision. Slot order remains stable
when a live peer takes over a failed controller. Ryu defaults to OpenFlow 1.5;
set `OPENFLOW_VERSION = "1.3"` in `eagle_controller.py` for OpenFlow 1.3.

Keep this directory named `ryu/`. Its package path includes the installed Ryu
library; the local app uses `ryu.eagle_controller` so upstream `ryu.controller`
remains available. The package exposes the installed Ryu version for its manager.

The supervisor publishes its Unix socket address in `.socket` at the repository
root. EAGLE discovers it or accepts `--ryu-socket`, and restores stopped managed
controllers before trials. Changing topology requires restarting both services.

Use `make controllers ARGS='--network networks/v0.json'` to launch, and
`make test-ryu` for unit tests. See the [repository overview](../README.md) and
[workflow notebook](../summary.ipynb) for experiment commands.
