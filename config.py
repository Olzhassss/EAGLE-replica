class Config:
    SEED = 42
    DECISION_SECONDS = 5
    MIN_DECISIONS = 20
    TRAFFIC_SCALE_FACTOR = None
    """Time stretch for generated traffic. None uses the number of hosts."""
    N_DOMAINS = 3
    """Number of domains (communities/subnets)"""
    SWITCHES_PER_DOMAIN = 1
    """Number of controllable switches in each domain."""
    LOUVAIN_CAPACITY = None
    """Move capacity, or the ceiling of switches per domain when unset."""
    LOUVAIN_DIFFERENCE = 2
    LOUVAIN_RHO = 0.01
    LOUVAIN_RESOLUTION = 0.1
