"""Collect raw FeatherGraph observations with every support switch enabled."""

from .artifacts import EVAL, MODELS
from .checkpoints import checkpoint_digest, load_inference_model
from .datasets import Dataset
from .evaluate import make_environment, seed_everything
from .run_log import RunLog
from .scenario import Scenario
from .trials import run_trial


def collect_embeddings(
    args,
    network: Scenario,
    settings: dict,
    dataset: Dataset,
    count: int,
    *,
    nonzero: bool = False,
) -> dict:
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
        lambda _: [1] * len(network.controller_ids),
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
