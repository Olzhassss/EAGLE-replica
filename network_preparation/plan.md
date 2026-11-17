Provide a function to generate random networks with configurable (through CLI) number of nodes and links. This will be the network of switches.
In addition, add a function to parse networks from ./topology-zoo into networks as well. This function should print the number of nodes and links for each parsed network.
Link capacity should be random between 58 and 100 Mbps (uniform distribution). All random operations are based on the seed in config.py.

Run modified Louvian on the network(s) with config-defined number of communities and other variables, then run controller placement algorithm on each community. Do not import from partitioning/.

Select which switches can be turned off (i.e. controllable switches) based on the config-defined number of switches. The RL system will be adapted for a varied action space from number of controllable switches instead of just the number of controllers later. Switches are selected based on the number of active links (most connected nodes). Ignore swiches that cannot be disabled because they split the network. If there happens to be not enough switches to satisfy the requirements, raise error. Controller can be placed on a controllable switch node.

Generate and attach hosts to the network based on a configurable (through CLI) number of hosts. Hosts should be uniformly distributed among communities as much as possible. Hosts should be attached to switches with the lowest number of links (peripheral). If there are not enough such switches, next switches in the hierarchy should be used (with higher number of links), until all switches have a host connected. Controllable switches cannot have hosts attached to them. If there are more hosts than non-controllable switches, raise error.

Generated/processed networks should be saved in `networks/`. The generated files will be used by RL agent and dataset preparation scripts.

After generating each network, it should also be visualized and visualization saved in `networks/` as well. Visualization needs to differentiate hosts from switches, show links with capacity in Mbps, show domains (communities), show controllable switches, and where controllers are located (on which swich nodes).
