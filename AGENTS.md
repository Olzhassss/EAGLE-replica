# Repository guidance

## Structure and runtimes

- Work from the repository root. `EAGLE/__main__.py` dispatches `python -m EAGLE`.
- `EAGLE/src/` owns preparation, models, training, experiments, communication,
  measurement, and evaluation. Use absolute imports under `EAGLE.src`.
- `mininet/` is the emulation service for Linux system Python 3.10+.
  `ryu/` is the OpenFlow service for a dedicated Python 3.7 environment.
  Keep the directory names `mininet/` and `ryu/` as requested by the user.
  Their `__init__.py` files extend the package paths to include installed libraries.
  Keep the local Ryu app in `ryu/eagle_controller.py`; `ryu.controller` belongs
  to the upstream library.
- Components share JSON and HTTP/Unix socket protocols. Do not import EAGLE's
  modern Python modules into the Ryu or Mininet processes.
- `network_schema.py` is the shared, dependency-free network/action contract.
  Keep it Python 3.7 compatible so all three runtimes can validate the same data.
- `settings/` contains runtime presets; `EAGLE/src/config.py` contains preparation
  constants; `EAGLE/src/paths.py` resolves repository inputs and outputs.

## Development and verification

```sh
uv sync --locked --project EAGLE
EAGLE/.venv/bin/python -m EAGLE --help
make test
make test-ryu
```

The Makefile accepts `PYTHON` and `RYU_PYTHON` overrides for existing environments.
Ryu setup is documented in `ryu/README.md`. Its code must remain
Python 3.7 compatible: use `typing` containers and `Optional`/`Union` there.
Mininet runner code must remain compatible with system Python 3.10+.

Tests live in `EAGLE/tests`, `mininet/tests`, and `ryu/tests`.
They use fixtures/mocks and a local Unix socket, without live controllers or
root. Run the affected suites, and exercise CLI help when changing entry points.
Validate notebook code cells without running experiments as part of routine
checks. Full emulation requires Linux, root, external captures, and running
services; unit tests do not establish live network correctness.

## Implementation contracts

- Prefer direct functions and small domain modules. Add shared helpers for real
  duplication; avoid compatibility shims, generic registries, and unused hooks.
- Preserve controller slot order across embeddings, actions, rewards, agent
  state, and checkpoints, including controller failover.
- Support-switch counts are configurable, including different counts per domain
  and zero. Each controller has an ordered `support_switches` list; bit i of its
  integer action enables list element i. Actors have 2**count outputs. Preserve
  list order, validate mask ranges, and count set bits in switch evaluations.
- Network manifests use schema version 2; traffic datasets retain version 1. Dataset host identities
  must be explicit and agree with the selected network. Include capture contents
  in dataset fingerprints; preserve network fingerprint checks across services.
- All-on resets, baselines, calibration, topology changes, routing, and interactive
  display must handle every configured support switch. Select supports so all
  disable combinations stay connected, and never attach hosts to supports.
- Store support counts in checkpoints and learner state. Reject incompatible
  target layouts before loading weights; shared reward estimators encode joint
  actions as per-switch bits, so peer estimator dimensions stay consistent.
- Collect telemetry only from fresh, synchronized windows after acknowledged
  actions. Keep action revisions and ownership checks intact.
- Policy training freezes the reward estimator after pretraining. Keep time-limit
  truncation distinct from termination for bootstrapping, and retain optimizer
  and action-RNG state in resumable checkpoints.
- Calibration bounds come from source training and travel with the checkpoint.
  Do not recalibrate them on evaluation datasets or target topologies.
- Measure flow completion from delivered payloads; account for retransmissions
  and retain null completion times for incomplete flows. Keep modeled power
  distinct from physical energy measurements.
- Keep trial cleanup, worker shutdown, atomic checkpoint writes, and output-name
  collision checks when simplifying execution paths.

## Artifacts and documentation

Generated datasets, networks, models, evaluation runs, captures, sockets, and
virtual environments are local artifacts. Preserve existing user artifacts
during code cleanup. Keep generated notebook outputs out of committed changes.
Update the affected component README when commands or responsibilities change;
keep the root README focused on the codebase and workflow overview.
