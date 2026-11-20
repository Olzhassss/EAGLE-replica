# Experiment settings

`default.json` defines decision/warmup/drain timing, telemetry timeouts, Feather
embedding dimensions and normalization, the throughput/power reward, episode
budgets, optimizer rates, pruning, and evaluation repetitions.

`no_pruning.json` and `static_pruning.json` override only the pruning mode.
`EAGLE.src.settings.load_settings` recursively merges a selected JSON object
into the defaults and validates names, section shapes, and numeric ranges.
Use `--settings PATH` with training and evaluation commands:

```sh
python -m EAGLE train --network v0.json --traffic synthetic --output dc_t0_no_pruning --settings settings/no_pruning.json
```

CLI episode arguments override their corresponding settings. Preparation
constants belong in `EAGLE/src/config.py`. Network polling intervals belong in
the generated network JSON. See [EAGLE](../EAGLE/README.md) for component details.

Support-switch counts are network configuration. Use preparation's
`--support-switches` option or the `SWITCHES_PER_DOMAIN` preparation default.
Training, inference, and evaluation derive their action dimensions from the
generated network's ordered support lists, rather than from runtime settings.
