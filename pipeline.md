# Train and evaluate EAGLE

Run from the repository root. Before each trial, Ryu and Mininet must be running with the exact same scenario JSON passed to the RL command. When the scenario changes, restart the launchers with that file, Ryu first. The RL CLI discovers both control sockets. Use fresh output directories.

The default settings use the study's method with example power and training values. For a reproduction, complete `settings/study.json` with recovered inputs and pass the resulting file via `--settings`. [Study inputs](scenarios/README.md#published-protocol)

## 1. Define the source training sequence

Paste this into the shell where you will run the remaining commands. It collects source-only calibration data, trains, then evaluates the frozen policy.

```bash
PY=model/.venv/bin/python
train_eval() {
  local scenario=$1 train_traffic=$2 eval_traffic=$3 out=$4
  "$PY" -m model.evaluate --scenario "$scenario" --traffic "$train_traffic" \
    --fixed 1,1,1 --identity-normalization --output "$out/calibration" &&
  "$PY" -m model.train --scenario "$scenario" --traffic "$train_traffic" \
    --normalize-from "$out/calibration/events.jsonl" --episodes 3 \
    --output "$out/train" &&
  "$PY" -m model.evaluate --scenario "$scenario" --traffic "$eval_traffic" \
    --checkpoint "$out/train/checkpoint.pth" --output "$out/eval"
}
```

The fixed action vector here assumes three logical controller slots. Adjust it for other source scenarios.

## 2. Run synthetic traffic on V0

With both launchers on `scenarios/v0.json`:

```bash
train_eval scenarios/v0.json synthetic synthetic runs/pipeline-synthetic
```

## 3. Train on Digital Corpora and evaluate on its held-out window

Prepare the source scenario. PCAP trials need `tcpreplay` and `tcpdump` on the Mininet host.

```bash
"$PY" -m model.prepare_traces 5gb-tcp-connection.pcap.gz --output runs/pipeline-corpora-data
```

With both launchers on `runs/pipeline-corpora-data/scenario.json`:

```bash
train_eval runs/pipeline-corpora-data/scenario.json pcap-train pcap-eval runs/pipeline-corpora
```

## 4. Evaluate the Corpora checkpoint on Megalodon

If the Ethernet PCAP prefix is already available, use it directly and skip extraction and conversion.

```bash
mkdir -p runs/pipeline-megalodon-input
7z e -oruns/pipeline-megalodon-input MegalodonChallenge.7z 'Content Server.pcapng'
tcpdump -n -r 'runs/pipeline-megalodon-input/Content Server.pcapng' -c 40000 \
  -w runs/pipeline-megalodon-input/content-first-40000.pcap
"$PY" -m model.prepare_traces runs/pipeline-megalodon-input/content-first-40000.pcap \
  --scenario runs/pipeline-corpora-data/scenario.json --recipe scenarios/megalodon-example.json \
  --output runs/pipeline-megalodon-data
```

With both launchers on `runs/pipeline-megalodon-data/scenario.json`:

```bash
"$PY" -m model.evaluate --scenario runs/pipeline-megalodon-data/scenario.json \
  --traffic megalodon-t2 --checkpoint runs/pipeline-corpora/train/checkpoint.pth \
  --output runs/pipeline-megalodon-eval
```

## 5. Evaluate on a different topology

Use a complete target scenario with the same controller IDs in the same order as the source checkpoint. For a runnable example, create a V0 variant that removes the `s9–s10` link. Write it next to the source JSON so the replay paths still resolve.

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

source = Path("runs/pipeline-megalodon-data/scenario.json")
scenario = json.loads(source.read_text())
scenario["id"] = "V0-minus-s9-s10"
scenario["provenance"] = "Example topology transfer from V0 with the s9-s10 link removed."
scenario["links"] = [
    link for link in scenario["links"]
    if {link["source"], link["target"]} != {"s9", "s10"}
]
source.with_name("target.json").write_text(json.dumps(scenario, indent=2) + "\n")
PY
```

With both launchers on `runs/pipeline-megalodon-data/target.json`, evaluate each traffic type using its source checkpoint.

```bash
TARGET=runs/pipeline-megalodon-data/target.json
"$PY" -m model.evaluate --scenario "$TARGET" --traffic synthetic \
  --checkpoint runs/pipeline-synthetic/train/checkpoint.pth --output runs/pipeline-target-synthetic
"$PY" -m model.evaluate --scenario "$TARGET" --traffic pcap-eval \
  --checkpoint runs/pipeline-corpora/train/checkpoint.pth --output runs/pipeline-target-corpora
"$PY" -m model.evaluate --scenario "$TARGET" --traffic megalodon-t2 \
  --checkpoint runs/pipeline-corpora/train/checkpoint.pth --output runs/pipeline-target-megalodon
```

For a separate model trained on the target, run `train_eval "$TARGET" synthetic synthetic runs/pipeline-target-train`.
