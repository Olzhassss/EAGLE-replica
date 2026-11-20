"""Defaults for dataset generation and network partitioning.

Experiment timing, learning, and reward parameters belong in settings/*.json.
"""

SEED = 42
DECISION_SECONDS = 5
MIN_DECISIONS = 20
# None stretches capture windows by the number of hosts.
TRAFFIC_SCALE_FACTOR = None
N_DOMAINS = 3
# Uniform support-switch count; preparation's --support-switches can override per domain.
SWITCHES_PER_DOMAIN = 1
# None uses the ceiling of switches per domain as the Louvain move capacity.
LOUVAIN_CAPACITY = None
LOUVAIN_DIFFERENCE = 2
LOUVAIN_RHO = 0.01
LOUVAIN_RESOLUTION = 0.1
