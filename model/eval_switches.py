"""Compare enabled controllable switches over one fixed-length episode per policy."""

import argparse

from .artifacts import EVAL, MODELS, write_json
from .checkpoints import checkpoint_digest, load_inference_model
from .cli import add_runtime_arguments, artifact_name, load_network, resolve_device
from .datasets import load_dataset
from .evaluate import make_environment, seed_everything
from .plots import plot_switch_counts
from .run_log import RunLog
from .settings import load_settings, positive
from .trials import run_trial


def evaluate_switches(args: argparse.Namespace) -> dict:
    positive(args.steps, "steps", integer=True)
    positive(args.seed, "seed", integer=True, zero=True)
    if len(args.models) != 3 or len(set(args.models)) != 3 or "baseline" in args.models:
        raise ValueError("Choose three distinct model names. The baseline name is reserved")
    network = load_network(args.network)
    settings = load_settings(args.settings)
    dataset = load_dataset(network, args.dataset, "test")
    name = args.name or f"switches_{network.path.stem}"
    device = resolve_device(args.device)
    loaded = {model: load_inference_model(MODELS / model, network) for model in args.models}
    for policy, _ in loaded.values():
        policy.to(device)
    encoder = next(iter(loaded.values()))[1]
    methods = {"baseline": (lambda _: [1] * len(network.controller_ids), encoder)}
    methods.update({
        model: (policy.decide, saved_encoder)
        for model, (policy, saved_encoder) in loaded.items()
    })
    counts = {}
    for model, (decide, saved_encoder) in methods.items():
        seed_everything(args.seed)
        environment = make_environment(args, network, settings, saved_encoder)
        log = RunLog(
            EVAL / "runs" / name / network.path.stem / dataset.name / model,
            network, settings, overwrite=True, seed=args.seed, model=model,
            dataset=dataset.name, steps=args.steps, encoder=saved_encoder.to_dict(),
        )
        enabled = []

        def record(transition: dict) -> None:
            enabled.append(sum(transition["action"]))

        run_trial(
            environment, decide, log, episode=1, dataset=dataset, steps=args.steps,
            on_transition=record, progress=model, measure_flows=False,
        )
        if len(enabled) != args.steps:
            raise ValueError(f"{model} completed {len(enabled)} of {args.steps} requested steps")
        counts[model] = enabled
    report = {
        "network": network.path.name, "network_sha256": network.fingerprint,
        "dataset": dataset.name, "dataset_sha256": dataset.fingerprint,
        "dataset_duration_seconds": dataset.duration,
        "models_sha256": {model: checkpoint_digest(MODELS / model) for model in args.models},
        "seed": args.seed, "steps": args.steps, "settings": settings,
        "enabled_switches": counts,
        "final_enabled_switches": {model: enabled[-1] for model, enabled in counts.items()},
    }
    write_json(EVAL / f"{name}.json", report)
    plot_switch_counts(counts, EVAL / f"{name}.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--network", required=True, help="Filename in networks/")
    parser.add_argument("--steps", type=int, required=True, help="Joint decisions per episode")
    parser.add_argument(
        "--models", nargs=3, type=artifact_name, default=["dc_t0", "dc_pcaps", "dc_full"],
    )
    parser.add_argument(
        "--dataset", type=artifact_name, default="dc_t0", help="Prepared test manifest stem, e.g. dc_t0",
    )
    parser.add_argument("--name", type=artifact_name, help="Output prefix in eval/ (default switches_<network>)")
    args = parser.parse_args()
    try:
        evaluate_switches(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
