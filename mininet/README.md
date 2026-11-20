# Mininet runner

This service builds a generated network and runs prepared traffic in its host
namespaces. Install Mininet, Open vSwitch, `tcpreplay`, and `tcpdump` for Linux
system Python 3.10+. Start [Ryu](../ryu/README.md) first with the same
network JSON, then run from the repository root:

```sh
sudo python3 -m mininet --network networks/v0.json --cli
```

Omit `--cli` to run only the control service. The launcher enables OpenFlow 1.3
and 1.5, applies configured ports and Mbps capacities, and installs static ARP.
Keep this directory named `mininet/`. Its package path includes the system
Mininet library so imports such as `mininet.net` remain available.

Generated network JSON uses schema version 2. The shared `network_schema.py`
validates support-switch ownership, list uniqueness, initial masks, and fixed
host attachments before Mininet creates nodes. Ryu applies support-switch masks;
Mininet continues to build all configured switches and links.

| File                | Responsibility                                                              |
| ------------------- | --------------------------------------------------------------------------- |
| `network.py`        | Validate command-facing network values and create nodes/links               |
| `launch.py`         | Own Mininet, the socket server, and shutdown cleanup                        |
| `control_server.py` | Serve `status`, `reset`, `traffic`, and `finish` commands                   |
| `traffic.py`        | Validate jobs, schedule processes, capture traffic, and drain/stop trials   |
| `traffic_flow.py`   | Shared asynchronous TCP receivers, scheduled senders, and timed PCAP replay |
| `tests/`            | Network validation and traffic lifecycle checks without live Mininet        |

The launcher writes its Unix socket address to `mininet/.socket`.
EAGLE discovers this file or accepts `--mininet-socket`. Commands carry
newline-delimited JSON and are restricted to the session owner or root.
Synthetic traffic records sender starts and receiver completions. PCAP jobs
preserve prepared timing and optionally capture both ends for payload analysis.

Ryu controls forwarding actions. A topology change requires restarting both
services. Ctrl-C or exiting the CLI stops traffic, closes Mininet, and removes
the address file. If forced termination leaves that file behind, verify the
old process has stopped before removing it and relaunching.

`make network ARGS='--network networks/v0.json --cli'` runs the launcher.
`make test` includes this component's unit tests. See the
[repository overview](../README.md) for the full workflow and output locations.
