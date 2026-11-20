"""Compare enabled controllable switches over one fixed-length episode per policy."""

import argparse

from EAGLE.src.arguments import add_runtime_arguments, artifact_name, validate_models
from EAGLE.src.artifacts import write_json
from EAGLE.src.datasets import load_dataset
from EAGLE.src.evaluation.plots import plot_switch_counts
from EAGLE.src.evaluation.policy_comparison import load_policies
from EAGLE.src.experiments.run_log import RunLog
from EAGLE.src.experiments.runtime import make_environment, seed_everything
from EAGLE.src.experiments.trials import run_trial
from EAGLE.src.model.checkpoints import checkpoint_digest
from EAGLE.src.network import load_network
from EAGLE.src.paths import EVAL, MODELS
from EAGLE.src.settings import load_settings, positive


def evaluate_switches(args: argparse.Namespace) -> dict:
    """Run each frozen policy and the baseline for an identical decision horizon."""
    positive(args.steps, "steps", integer=True)
    positive(args.seed, "seed", integer=True, zero=True)
    validate_models(args.models)
    network = load_network(args.network)
    settings = load_settings(args.settings)
    dataset = load_dataset(network, args.dataset, "test")
    name = args.name or f"switches_{network.path.stem}"
    methods = load_policies(args.models, network, args.device)
    counts = {}
    for model, (decide, saved_encoder) in methods.items():
        seed_everything(args.seed)
        environment = make_environment(args, network, settings, saved_encoder)
        log = RunLog(
            EVAL / "runs" / name / network.path.stem / dataset.name / model,
            network,
            settings,
            overwrite=True,
            seed=args.seed,
            model=model,
            dataset=dataset.name,
            steps=args.steps,
            encoder=saved_encoder.to_dict(),
        )
        enabled = []

        def record(transition: dict) -> None:
            enabled.append(sum(int(mask).bit_count() for mask in transition["action"]))

        run_trial(
            environment,
            decide,
            log,
            episode=1,
            dataset=dataset,
            steps=args.steps,
            on_transition=record,
            progress=model,
            measure_flows=False,
        )
        if len(enabled) != args.steps:
            raise ValueError(
                f"{model} completed {len(enabled)} of {args.steps} requested steps"
            )
        counts[model] = enabled
    report = {
        "network": network.path.name,
        "network_sha256": network.fingerprint,
        "support_switches": {
            row["id"]: row["support_switches"] for row in network.data["controllers"]
        },
        "action_sizes": list(network.action_sizes),
        "dataset": dataset.name,
        "dataset_sha256": dataset.fingerprint,
        "dataset_duration_seconds": dataset.duration,
        "models_sha256": {
            model: checkpoint_digest(MODELS / model) for model in args.models
        },
        "seed": args.seed,
        "steps": args.steps,
        "settings": settings,
        "enabled_switches": counts,
        "final_enabled_switches": {
            model: enabled[-1] for model, enabled in counts.items()
        },
    }
    write_json(EVAL / f"{name}.json", report)
    plot_switch_counts(counts, EVAL / f"{name}.png")
    return report


def main() -> None:
    """Parse the switch-count experiment and save its report and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--network", required=True, help="Filename in networks/")
    parser.add_argument(
        "--steps", type=int, required=True, help="Joint decisions per episode"
    )
    parser.add_argument(
        "--models",
        nargs=3,
        type=artifact_name,
        default=["dc_t0", "dc_pcaps", "dc_full"],
    )
    parser.add_argument(
        "--dataset",
        type=artifact_name,
        default="dc_t0",
        help="Prepared test manifest stem, e.g. dc_t0",
    )
    parser.add_argument(
        "--name",
        type=artifact_name,
        help="Output prefix in eval/ (default switches_<network>)",
    )
    args = parser.parse_args()
    evaluate_switches(args)


if __name__ == "__main__":
    main()
