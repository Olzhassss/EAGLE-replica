"""Allocate pretraining and policy episodes to traffic conditions."""

from EAGLE.src.settings import positive


def training_stages(mode: str, episodes: int, pretrain_episodes: int) -> list[dict]:
    """Pretrain across the source conditions, then freeze the estimator."""
    positive(episodes, "episodes", integer=True)
    positive(pretrain_episodes, "pretrain_episodes", integer=True)
    profiles = ["dc_pcap"] if mode == "pcaps" else ["dc_t0"]
    if mode == "curriculum":
        profiles += [f"dc_t0_trenv{index}" for index in range(2, 6)]
    if mode not in ("pcaps", "synthetic", "curriculum"):
        raise ValueError("Training mode must be pcaps, synthetic, or curriculum")
    if episodes < len(profiles):
        raise ValueError("Curriculum training needs at least five policy episodes")
    stages = [
        {
            "name": "pretrain",
            "mode": "pretrain",
            "profiles": profiles,
            "episodes": pretrain_episodes,
        }
    ]
    share, extra = divmod(episodes, len(profiles))
    stages.extend(
        {
            "name": name,
            "mode": "frozen",
            "profiles": [name],
            "episodes": share + (index < extra),
        }
        for index, name in enumerate(profiles)
    )
    return stages
