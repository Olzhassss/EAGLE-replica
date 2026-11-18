# Ryu OpenFlow controllers

Create the dedicated Python 3.7 environment from the repository root.

```sh
python3.7 -m venv ryu_controller/.venv
ryu_controller/.venv/bin/python -m pip install 'setuptools==57.5.0' wheel
ryu_controller/.venv/bin/python -m pip install --no-build-isolation -r ryu_controller/requirements.txt
ryu_controller/.venv/bin/python ryu_controller/launch_controllers.py --scenario networks/v0.json
```

Start Ryu before Mininet with the identical generated network JSON. The launcher uses its Python 3.7 interpreter for each child. Ctrl-C shuts down those children. `--controller ID` selects controllers for a separately placed deployment with reachable network endpoints. Synchronize clocks across hosts for sample-window and action-revision checks.

Ryu defaults to OpenFlow 1.5. Set `OPENFLOW_VERSION = "1.3"` in `controller.py` when using OpenFlow 1.3. The Mininet launcher enables both versions on its bridges.

`controller.py` owns OpenFlow and HTTP handlers, port validation, master ownership, and route reconciliation. `topology_state.py` maintains configured, active, and observed graphs while preserving logical controller slots. `routing.py` computes stable end-to-end paths per ordered host MAC pair, preferring the source domain's support switch among equal-cost choices. `openflow_rules.py` reconciles changed EAGLE-owned rules. `telemetry.py` pairs full poll replies, handles counter resets, and attributes delivered bytes to destination attachment switches.

| Interface | Purpose |
| --- | --- |
| `GET /network/state` | Controller topology, slot ownership, and action revisions |
| `GET /network/stats?controller_id=c0` | Fresh coherent telemetry, or HTTP 503 while a sample is pending |
| `PUT /model/actions?controller_id=c0&details=1` | Apply `[0]` or `[1]` and acknowledge its revision and acceptance time |

The supervisor writes its address to `.socket` at the repository root. The model discovers it for restoring the controller processes before an episode. Use `--ryu-socket` to override that address. Restart the supervisor and Mininet manually when changing topology.

[Root documentation](../Documentation.md) contains the experiment commands, data contracts, and evaluation protocol.
