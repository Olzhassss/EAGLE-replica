# EAGLE replica

A research implementation of multi-agent reinforcement learning for SDN switch
autoscaling, inspired by Monaco et al., *Adaptive SDN Autoscaling via Generalizable
Multi-Agent Reinforcement Learning With EAGLE* (IEEE TNSM, 2026,
DOI `10.1109/TNSM.2026.3717361`). Initially forked from the
[authors' repository](https://github.com/dahara98/EAGLE).

## Codebase

| Component | Responsibility | Runtime |
| --- | --- | --- |
| [EAGLE](EAGLE/README.md) | Prepare inputs, train policies, run trials, and evaluate results | Python 3.13+ |
| [Mininet runner](mininet/README.md) | Build the emulated network and schedule traffic | Linux system Python 3.10+, Mininet, Open vSwitch; root for emulation |
| [Ryu controller](ryu/README.md) | Supervise controllers, program OpenFlow rules, and collect telemetry | Separate Python 3.7 environment |
| [Settings](settings/README.md) | Experiment timing, rewards, learning, and pruning presets | JSON |
| `network_schema.py` | Shared network schema, support-switch layout validation, and action-mask contract | Dependency-free Python 3.7+ |
| [Workflow notebook](summary.ipynb) | Guided preparation, training, post-training, and evaluation recipes | EAGLE notebook kernel |

The service directories remain `mininet/` and `ryu/`. Their package paths include
the installed libraries, so local service modules and upstream modules can be
imported together. The three components share network JSON and communicate through
HTTP and local Unix sockets; they do not share a Python environment.

## Workflow

1. Prepare packet captures into traffic manifests with explicit host addresses.
2. Generate a network JSON using the matching dataset's host metadata.
3. Start Ryu, then Mininet, in separate terminals with the same network JSON.
4. Train or apply a policy. EAGLE sends joint actions through HTTP, waits for
   acknowledged telemetry windows, and measures delivered traffic.
5. Evaluate frozen policies against the all-on baseline and inspect saved reports.

Each domain has one agent and a configurable ordered list of support switches.
With `k` support switches, its action space contains `2^k` on/off combinations.
See [EAGLE's configuration examples](EAGLE/README.md#support-switch-configuration).
Network definitions use schema version 2; traffic datasets retain version 1.

Install the experiment environment and inspect its commands from the repository root:

```sh
uv sync --locked --project EAGLE
uv run --project EAGLE python -m EAGLE --help
```

Follow the component READMEs for preparation, controller setup, and launch commands.
Restart both services when changing network definitions. The notebook assumes
that service restarts happen in separate terminals.

## Inputs and outputs

| Location | Contents |
| --- | --- |
| `EAGLE/src/network_preparation/` | Bundled source topology and Topology Zoo GraphML inputs |
| `datasets/<N>_hosts/{train,test}/` | Generated traffic manifests, captures, and charts |
| `networks/` | Generated network definitions and SVG plots |
| `models/` | Inference models and training reward histories |
| `models/checkpoints/` | Resumable learner state and training run logs |
| `eval/` | Evaluation reports, figures, and per-trial logs |

External source captures and generated artifacts are excluded from version control.
Repository paths are defined in `EAGLE/src/paths.py`; command arguments may supply
explicit input paths. Run the commands and notebook from the repository root.

## Checks

```sh
make test       # EAGLE and Mininet runner unit tests
make test-ryu   # Ryu tests in its Python 3.7 environment
```

The tests use mocks and local fixtures, including a Unix socket server. Full
OpenFlow and traffic experiments require the running services described above.
`PYTHON` and `RYU_PYTHON` override the interpreter paths used by the Makefile.
