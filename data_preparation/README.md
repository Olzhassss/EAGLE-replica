# Data preparation

Run both commands from the repository root with the RL Python environment.

```bash
model/.venv/bin/python -m data_preparation.prepare default \
  --hosts 9

model/.venv/bin/python -m data_preparation.prepare congestion \
  --hosts 9 --network <networks/something>
```

TODO: remove scenario

```bash
# output graphs for digital corpora. Assumes 5gb-tcp-connection.pcap.gz exists in the root dir.
model/.venv/bin/python -m data_preparation.analyze_trace
# output graphs for megalodon. Assumes megalodon.pcapng exists in the root dir.
model/.venv/bin/python -m data_preparation.analyze_trace --mg
```
