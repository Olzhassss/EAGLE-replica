"""Collect raw FeatherGraph observations with every support switch enabled."""

from EAGLE.src.datasets import Dataset
from EAGLE.src.experiments.run_log import RunLog
from EAGLE.src.experiments.runtime import make_environment, seed_everything
from EAGLE.src.experiments.trials import run_trial
from EAGLE.src.model.checkpoints import checkpoint_digest, load_inference_model
from EAGLE.src.network import Network
from EAGLE.src.paths import EVAL, MODELS


def collect_embeddings(
    args,
    network: Network,
    settings: dict,
    dataset: Dataset,
    count: int,
    *,
    nonzero: bool = False,
) -> dict:
    """Collect all-on graph observations using a checkpoint encoder and fixed seed."""
    if dataset.decisions(settings) < count:
        raise ValueError(f"{dataset.name} is shorter than {count} decision intervals")
    seed_everything(args.seed)
    path = MODELS / args.model
    _, encoder = load_inference_model(path, network)
    environment = make_environment(args, network, settings, encoder)
    log = RunLog(
        EVAL / "runs" / args.name / network.path.stem / dataset.name,
        network,
        settings,
        overwrite=True,
        model=args.model,
        seed=args.seed,
    )
    result = run_trial(
        environment,
        lambda _: network.all_on_actions,
        log,
        episode=1,
        dataset=dataset,
        observations=count,
        nonzero_only=nonzero,
        measure_flows=False,
    )
    return {
        "model": args.model,
        "model_sha256": checkpoint_digest(path),
        "network": network.path.stem,
        "network_sha256": network.fingerprint,
        "dataset": dataset.name,
        "dataset_sha256": dataset.fingerprint,
        "seed": args.seed,
        "encoder": encoder.to_dict(),
        "embeddings": result["embeddings"],
    }
