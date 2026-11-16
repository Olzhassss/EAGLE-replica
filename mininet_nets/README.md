# Mininet emulation

Install Mininet and Open vSwitch for the host's system Python 3. Use `tcpreplay` and `tcpdump` for prepared PCAP jobs. The built-in synthetic TCP transfer uses the Python standard library.

## Launch and shut down

From the repository root, start Ryu first, then run:

```sh
sudo python3 mininet_nets/launch_topology.py --scenario scenarios/v0.json
```

The launcher starts the scenario's OpenFlow 1.3 bridges, sets the configured switch datapath IDs and port numbers, installs static host ARP entries, and writes the Unix control socket address to `mininet_nets/.socket`. From the repository root, pass it to the RL project with `--mininet-socket "$(cat mininet_nets/.socket)"`. The socket accepts commands from the user who started Mininet through sudo. Add `--cli` to inspect the topology interactively. Exit the CLI or use Ctrl-C to shut down managed traffic and Mininet and remove the address file. **If a forced termination leaves the file behind, remove it before restarting Mininet.**

## Traffic and physical actions

- The `idle` profile has no sender, and the `synthetic` profile starts a bounded TCP transfer between configured host namespaces. The supervisor records the source start, receiver byte count, and completion time.
- A `pcap` job references a pre-cut one-way TCP Ethernet capture with addresses rewritten for the chosen source and destination hosts. The manifest supplies the rate, start time, and SHA-256. The runner captures both transmitting and receiving interfaces for delivery analysis. Use [`model.prepare_traces`](../scenarios/README.md#run-the-digital-corpora-capture) to generate the replay files and scenario from the Digital Corpora gzip. Use one job per ordered host pair in a profile so the captures stay attributable.
- The default resource mode is `forwarding`. For physical overhead measurements, `interfaces` toggles the configured support links and `ovs` deletes and recreates the supporting bridge with the original datapath ID and ports. Fault events can also toggle individual scenario links. A new trial restores the configured physical network and clears previous failures.
- The launcher retains traffic files for its session, and the RL option `--keep-captures` archives them under the run directory. Resource changes and traffic are limited to configured emulation resources.

A useful smoke run is the [fixed policy in the root guide](../README.md#start-an-experiment). The run should yield fresh telemetry from all three slots and a synthetic flow record after an eight-step V0 trial.
