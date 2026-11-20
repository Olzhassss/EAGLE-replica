# Mininet emulation

Install Mininet, Open vSwitch, `tcpreplay`, and `tcpdump` for system Python 3.10+. Run from the repository root after starting Ryu with the same network.

```sh
sudo python3 mininet_nets/launch_topology.py --scenario networks/v0.json
```

The launcher builds the configured datapath IDs, ports, host addresses, and Mbps capacities, enables OpenFlow 1.3 and 1.5, and installs static host ARP entries. It writes the control address to `mininet_nets/.socket`.
The Unix socket accepts network status, reset, prepared traffic jobs, and traffic finish operations. Synthetic jobs use a shared TCP receiver and a scheduled sender process per participating host. PCAP jobs preserve prepared timestamps and capture transmitting and receiving interfaces. After the configured drain period, completion analysis uses successfully delivered payloads.

Actions change forwarding through Ryu. A topology change requires stopping both launchers and starting Ryu first with the new file. Add `--cli` to inspect the emulated network. Ctrl-C or exiting the CLI stops managed traffic and Mininet and removes the address file. Remove a stale address file after a forced termination before restarting.

[Root documentation](../Documentation.md) contains dataset generation, training, evaluations, and measurement definitions.
