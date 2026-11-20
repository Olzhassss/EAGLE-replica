"""Train named EAGLE models on prepared traffic and save resumable state."""

import argparse
import fcntl
import time

from .artifacts import MODELS, save_rewards
from .checkpoints import checkpoint_digest, load_checkpoint, save_checkpoint
from .cli import add_runtime_arguments, artifact_name, load_network, resolve_device
from .datasets import load_dataset
from .evaluate import make_environment, seed_everything
from .observations import ObservationEncoder, make_encoder
from .run_log import RunLog
from .settings import load_settings, positive, training_stages
from .trials import run_trial
from .workers import AgentPool


def train(args: argparse.Namespace) -> None:
    directory = MODELS / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{args.output}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Training {args.output} is already running") from error
        _train(args)


def _train(args: argparse.Namespace) -> None:
    network = load_network(args.network)
    settings = load_settings(args.settings)
    if args.episodes is not None:
        settings["training"]["episodes"] = args.episodes
    if args.pretrain_episodes is not None:
        settings["training"]["pretrain_episodes"] = args.pretrain_episodes
    source_name = getattr(args, "from_checkpoint", None)
    dataset_name = getattr(args, "dataset", None)
    if bool(source_name) != bool(dataset_name):
        raise ValueError("Post-training requires both --from-checkpoint and --dataset")
    source = None
    if source_name:
        if source_name == args.output:
            raise ValueError(
                "Post-training requires a new output name to preserve its source checkpoint"
            )
        source_path = MODELS / "checkpoints" / f"{source_name}.pth"
        source = load_checkpoint(source_path)
        if source.get("scenario_sha256") != network.fingerprint:
            raise ValueError("Post-training requires the source checkpoint's network")
        for key in (
            "embedding",
            "learning",
            "reward",
            "decision_seconds",
            "warmup_seconds",
        ):
            if source["settings"][key] != settings[key]:
                raise ValueError(
                    f"Post-training requires the source checkpoint's {key} settings"
                )
        pretraining = sum(
            stage["episodes"] for stage in source["stages"] if stage["mode"] == "pretrain"
        )
        if source["completed_episodes"] < pretraining:
            raise ValueError("Complete reward-estimator pretraining before post-training")
        positive(settings["training"]["episodes"], "episodes", integer=True)
        stages = [
            {
                "name": dataset_name,
                "mode": "frozen",
                "profiles": [dataset_name],
                "episodes": settings["training"]["episodes"],
            }
        ]
    else:
        stages = training_stages(
            args.traffic,
            settings["training"]["episodes"],
            settings["training"]["pretrain_episodes"],
        )
    datasets = {
        name: load_dataset(network, name, "train")
        for stage in stages
        for name in stage["profiles"]
    }
    for dataset in datasets.values():
        dataset.decisions(settings)
    checkpoint_path = MODELS / "checkpoints" / f"{args.output}.pth"
    run_directory = MODELS / "checkpoints" / f"{args.output}.run"
    model_path = MODELS / args.output
    identity = {
        "settings": settings,
        "stages": stages,
        "seed": args.seed,
        "scenario_sha256": network.fingerprint,
        "training_mode": "posttrain" if source else args.traffic,
        "datasets": {name: dataset.fingerprint for name, dataset in datasets.items()},
    }
    if source:
        identity.update(
            source_checkpoint={
                "name": source_name,
                "sha256": checkpoint_digest(source_path),
            },
        )
    resume = load_checkpoint(checkpoint_path) if args.resume else None
    if resume and any(resume.get(key) != value for key, value in identity.items()):
        raise ValueError(
            "Resume requires the same network, datasets, settings, seed, and training mode"
        )
    existing_outputs = (model_path, checkpoint_path, MODELS / f"{args.output}.csv")
    if not args.resume and any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            "Model name already exists. Use --resume or choose another name"
        )
    device = resolve_device(args.device)
    seed_everything(args.seed)
    print(f"Training {args.output} on {device}", flush=True)
    encoder = ObservationEncoder(
        network.controller_ids,
        settings["embedding"]["order"],
        settings["embedding"]["eval_points"],
    )
    initial_state = resume or source
    if initial_state:
        encoder = ObservationEncoder.from_dict(initial_state["encoder"])
    elif settings["embedding"]["normalization"] == "source":
        calibration = []
        environment = make_environment(args, network, settings, encoder)
        log = RunLog(
            run_directory / "calibration",
            network,
            settings,
            overwrite=True,
            seed=args.seed,
        )
        for dataset in datasets.values():
            for episode in range(1, settings["training"]["calibration_episodes"] + 1):
                result = run_trial(
                    environment,
                    lambda _: [1] * len(network.controller_ids),
                    log,
                    episode=episode,
                    dataset=dataset,
                    progress=f"Calibration {dataset.name}",
                    measure_flows=False,
                )
                calibration.extend(result["embeddings"])
        encoder = make_encoder(network, settings, calibration)
    log = RunLog(
        run_directory,
        network,
        settings,
        resume=bool(resume),
        overwrite=not bool(resume),
        seed=args.seed,
        stages=stages,
        encoder=encoder.to_dict(),
        source_checkpoint=identity.get("source_checkpoint"),
    )
    rewards = list(resume["episode_rewards"]) if resume else []
    completed = resume["completed_episodes"] if resume else 0
    round_number = initial_state["round"] if initial_state else 0
    total = sum(stage["episodes"] for stage in stages)
    started = time.monotonic()
    with AgentPool(
        encoder,
        settings["learning"],
        args.seed,
        distributed=args.workers == "process",
        device=device,
    ) as agents:
        if initial_state:
            agents.load_state_dict(initial_state["agents"])
        environment = make_environment(args, network, settings, encoder)
        environment.encode_local = agents.embed

        def checkpoint(episode: int) -> None:
            save_checkpoint(
                checkpoint_path,
                dict(
                    identity,
                    encoder=encoder.to_dict(),
                    agents=agents.state_dict(),
                    round=round_number,
                    completed_episodes=episode,
                    episode_rewards=rewards,
                ),
            )

        checkpoint(completed)
        episode = 0
        for stage in stages:
            for stage_episode in range(stage["episodes"]):
                episode += 1
                if episode <= completed:
                    continue
                dataset = datasets[
                    stage["profiles"][stage_episode % len(stage["profiles"])]
                ]
                batch = []

                def update() -> None:
                    nonlocal round_number
                    round_number += 1
                    transitions = {
                        field: [row[field] for row in batch]
                        for field in (
                            "state",
                            "action",
                            "reward",
                            "next_state",
                            "terminated",
                        )
                    }
                    metrics = agents.run_round(round_number, stage["mode"], transitions)
                    log.write("learning", episode=episode, stage=stage["name"], **metrics)
                    batch.clear()

                def collect(transition: dict) -> None:
                    batch.append(transition)
                    if len(batch) == settings["learning"]["batch_size"]:
                        update()

                result = run_trial(
                    environment,
                    agents.decide,
                    log,
                    episode=episode,
                    dataset=dataset,
                    on_transition=collect,
                    measure_flows=False,
                    progress=f"{stage['name']} episode {stage_episode + 1}/{stage['episodes']}",
                )
                if batch:
                    update()
                if stage["mode"] == "frozen":
                    rewards.append(result["cumulative_reward"])
                checkpoint(episode)
                save_rewards(MODELS / f"{args.output}.csv", rewards, plot=False)
                average_seconds = (time.monotonic() - started) / (episode - completed)
                remaining = total - episode
                print(
                    f"Episode {episode}/{total} | cumulative reward {result['cumulative_reward']:.6f} | "
                    f"remaining {remaining} | ETA {remaining * average_seconds / 60:.1f} min",
                    flush=True,
                )
        state = agents.state_dict()
        save_checkpoint(
            model_path,
            {
                "schema_version": 1,
                "encoder": encoder.to_dict(),
                "settings": settings,
                "agents": {
                    slot: {key: values[key] for key in ("actor", "critic")}
                    for slot, values in state.items()
                },
            },
        )
    save_rewards(MODELS / f"{args.output}.csv", rewards)
    print(f"Saved {model_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--network", required=True, help="Filename in networks/")
    traffic = parser.add_mutually_exclusive_group(required=True)
    traffic.add_argument("--traffic", choices=("pcaps", "synthetic", "curriculum"))
    traffic.add_argument(
        "--dataset", type=artifact_name, help="Prepared dataset name for post-training"
    )
    parser.add_argument(
        "--from-checkpoint",
        type=artifact_name,
        help="Initialize post-training from models/checkpoints/<name>.pth",
    )
    parser.add_argument(
        "--output",
        type=artifact_name,
        required=True,
        help="Inference filename in models/",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--pretrain-episodes", type=int)
    parser.add_argument(
        "--resume", action="store_true", help="Resume this model's latest checkpoint"
    )
    parser.add_argument("--workers", choices=("local", "process"), default="process")
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
