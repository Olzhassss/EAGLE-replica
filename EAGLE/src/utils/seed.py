def seed_everything(seed: int) -> None:
    positive(seed, "seed", integer=True, zero=True)
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
