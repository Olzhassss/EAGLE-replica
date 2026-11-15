"""Feather graph fingerprints and checkpoint-compatible observation scaling.

Based on the graph-level FEATHER algorithm of Rozemberczki and Sarkar. The
legacy calculation follows the checkout's Karate Club degree/loop conventions.
"""

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
from scipy.sparse import diags


LEGACY_MINIMUM = [2705, 2706, 41, -1, 2751, 2751, 42, -1, 2705, 2706, 41, -1]
LEGACY_MAXIMUM = [4969, 4970, 82, 1, 4542, 4543, 75, 1, 4918, 4919, 81, 1]


def embed_subnet(topology, *, order, eval_points, legacy=False):
    """Return a vector of 4 * order * eval_points graph features."""
    nodes = sorted(row["id"] for row in topology["nodes"])
    if not nodes:
        return np.zeros(4 * order * eval_points, dtype=np.float64)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for row in topology["links"]:
        graph.add_edge(row["source"], row["target"], weight=float(row.get("weight", 1.0)))
    graph.add_weighted_edges_from((node, node, 1.0) for node in nodes)

    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", dtype=float)
    if legacy:
        degree = np.array([graph.degree(node) for node in nodes], dtype=float)
        walk_degree = degree
    else:
        degree = np.array([graph.degree(node, weight="weight") for node in nodes], dtype=float)
        walk_degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    walk = diags(1.0 / walk_degree) @ adjacency
    features = np.column_stack((np.log1p(degree), [nx.clustering(graph, node) for node in nodes]))
    angles = (features[:, :, None] * np.linspace(0.01, 2.5, eval_points)).reshape(len(nodes), -1)
    features = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
    pooled = []
    for _ in range(order):
        features = walk @ features
        pooled.append(np.asarray(features).mean(axis=0))
    return np.concatenate(pooled)


@dataclass(frozen=True)
class ObservationEncoder:
    controller_ids: tuple[str, ...]
    order: int
    eval_points: int
    legacy: bool
    minimum: list[float] | None = None
    maximum: list[float] | None = None

    def __post_init__(self):
        if (self.minimum is None) != (self.maximum is None):
            raise ValueError("Specify both normalization bounds")
        if self.minimum is not None:
            if len(self.minimum) != self.size or len(self.maximum) != self.size:
                raise ValueError("Normalization dimensions do not match the model")
            if not np.isfinite(self.minimum + self.maximum).all():
                raise ValueError("Normalization bounds must be finite")
            if any(high < low for low, high in zip(self.minimum, self.maximum)):
                raise ValueError("Maximum must be at least minimum for every feature")

    @property
    def size(self):
        return 4 * self.order * self.eval_points * len(self.controller_ids)

    def embed_local(self, topology):
        vector = embed_subnet(topology, order=self.order, eval_points=self.eval_points, legacy=self.legacy)
        if self.legacy:
            return np.array([round(value, 4) * 10000 for value in vector])
        return vector

    def normalize(self, raw):
        """Scale the concatenated vector using bounds from the source checkpoint."""
        raw = np.asarray(raw)
        if raw.shape != (self.size,):
            raise ValueError("Expected one embedding per controller in slot order")
        if self.minimum is None:
            return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        lo = np.asarray(self.minimum)
        widths = np.asarray(self.maximum) - lo
        scaled = np.divide((raw - lo) * 100, widths, out=np.zeros(self.size), where=widths > 0)
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data, controller_ids=tuple(data["controller_ids"])))


def make_encoder(scenario, settings, calibration=()):
    config = settings["embedding"]
    legacy = settings["observation_view"] == "configured"
    encoder = ObservationEncoder(scenario.controller_ids, config["order"], config["eval_points"], legacy)
    if config["normalization"] == "legacy":
        if scenario.controller_ids != ("c0", "c1", "c2") or not legacy or encoder.size != 12:
            raise ValueError("Legacy min/max values require the original c0,c1,c2 observation contract")
        return ObservationEncoder(**dict(encoder.to_dict(), minimum=LEGACY_MINIMUM, maximum=LEGACY_MAXIMUM))
    if config["normalization"] == "source":
        bounds = None
        count = 0
        for vector in calibration:
            row = np.asarray(vector)
            if row.shape != (encoder.size,) or not np.isfinite(row).all():
                raise ValueError("Source calibration has an incompatible embedding")
            bounds = (row, row) if bounds is None else (np.minimum(bounds[0], row), np.maximum(bounds[1], row))
            count += 1
        if count < 2:
            raise ValueError("Source normalization requires at least two saved source embeddings")
        return ObservationEncoder(**dict(encoder.to_dict(), minimum=bounds[0].tolist(), maximum=bounds[1].tolist()))
    return encoder
