There are 2 traces -- Digital Corpora (DC) and Megalodon.

# DC

DC spans ~1871 seconds (rounded down) --> 1871 1s windows. Avg bps ~27062782
Only 2 hosts, most traffic from h1 to h2. So, use cumulative traffic stats to generate a dataset:

1. Calculate avg traffic stats per each 1s window, giving 1870 instances. Keep the list of original packet captures contained within each instance.
2. Randomly (using a predefined seed) split instances into 80% train and 20% test sets. Split by 1s windows.
3. Then randomly pool without repetition from train and test sets to X subsets to reach at least 12 episode steps or more. This value should be calculated from a configurable number of hosts `n` (depends on the target number of hosts in the actual network). If there is less data that necessary, pool with repetition. Pool by 1s windows.

Now, use this data to generate train and test datasets (1) using PCAPs and (2) for synthetic traffic.

For the PCAPs, merge all windows belonging to each directed host pair, resulting in train and test trace files for every directed host pair. While constructing datasets, set proper source and destination, as well as timestamp. Average stats for each window are not used.

For synthetic traffic, ignore original packets and only use average stats, resulting in train and test datasets that should be runnable using mininet (current implementation will be updated later).

Output location (relative to repo root):

1. PCAP train: `datasets/<n>_hosts/train/dc_pcap.<extension>`
2. PCAP test: `datasets/<n>_hosts/test/dc_pcap.<extension>`
3. Synthetic train: `datasets/<n>_hosts/train/dc_t0.<extension>`
4. Synthetic test: `datasets/<n>_hosts/test/dc_t0.<extension>`

# Megalodon

Megalodon has Content Server and Web App server traces, which are almost identical, so only Content Server trace will be used. Total duration is ~416 seconds (rounded down) -> 416 instances of windows. This trace will be used for evaluation only. Traffic from 192.168.1.1 to 172.28.190.238 should be considered separately, while the rest merged. To do that, pre-process the data so that traffic from 192.168.1.1 to 172.28.190.238 appears only after the rest stops, so that the total duration and number of instances is effectively doubled. The rest is the same as for DC. In the end, there should be 2 test datasets, one with PCAPs, the other for synthetic traffic.

Output location (relative to repo root):

2. PCAP test: `datasets/<n>_hosts/test/mg_pcap.<extension>`
3. Synthetic test: `datasets/<n>_hosts/test/mg_t0.<extension>`

# Global traffic load patterns

Having the default corpora dataset, we can generate traffic loads for evaluation:

0. Default DC test datasets (PCAP and synthetic)
1. Use default DC synthetic test dataset, but decrease the stats for each instance by a predefined constant scale factor (0.5)
2. Default Megalodon test datasets (PCAP and synthetic)
3. Use default DC synthetic test dataset, but increase the stats for each instance by a predefined constant scale factor (1.5)
4. Same as 3, but scale by a predifined const = 2.0.

Output location (relative to repo root):

1. Pattern 1: `datasets/<n>_hosts/test/dc_t1.<extension>`
2. Pattern 3: `datasets/<n>_hosts/test/dc_t3.<extension>`
3. Pattern 4: `datasets/<n>_hosts/test/dc_t4.<extension>`

# Regional traffic congestion setups

Given the partitioned network with hosts, we can generate congestion setups. Congestion setups will be available only for networks with 3 communities (assumed).
Setups:

1. Default (already done from previous steps)
2. Higher load in community 1.
3. Higher load in community 2.
4. Higher load in community 3.
5. Higher load in all communities.

To get higher load, we will use the same procedure that is used for modifying traffic load patters, but for both training and test datasets of default DC synthetic. Higher load in this context will mean a predefined constant scale factor of 1.5 for hosts in that community. With this scale factor, setup 5 is identical to Pattern 3, but it is ok.

Output location (relative to repo root):

1. Setup 1: already present at `datasets/<n>_hosts/train/dc_t0.<extension>` and `datasets/<n>_hosts/test/dc_t0.<extension>`
2. Setup 2: `datasets/<n>_hosts/train/dc_t0_trenv2.<extension>` and `datasets/<n>_hosts/test/dc_t0_trenv2.<extension>`
3. Setup 3: `datasets/<n>_hosts/train/dc_t0_trenv3.<extension>` and `datasets/<n>_hosts/test/dc_t0_trenv3.<extension>`
4. Setup 4: `datasets/<n>_hosts/train/dc_t0_trenv4.<extension>` and `datasets/<n>_hosts/test/dc_t0_trenv4.<extension>`
5. Setup 5: `datasets/<n>_hosts/train/dc_t0_trenv5.<extension>` and `datasets/<n>_hosts/test/dc_t0_trenv5.<extension>`

Since thus step requires a splitted network topology, it should be separate from the rest of the pipeline.

After generating, for each generated dataset there should be a chart visualizing throughput over 1s windows, specifying the value for each direction and a total. Visualizations should be saved in the same place where the final datasets are (skip tmp).
