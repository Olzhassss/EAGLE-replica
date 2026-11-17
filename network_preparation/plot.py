"""Render a zoomable map of domains, hosts, switch roles and link rates."""

from pathlib import Path

import networkx as nx


def save_network_plot(scenario: dict, path: Path, seed: int, *, pre_merge: bool = False) -> None:
    from matplotlib import colormaps, rc_context
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    controllers = [] if pre_merge else scenario["controllers"]
    if pre_merge:
        groups = scenario["provenance"]["pre_merge_communities"]
        owners = {node: f"L{index}" for index, group in enumerate(groups, 1) for node in group}
        domains = [f"L{index}" for index in range(1, len(groups) + 1)]
    else:
        owners = {row["id"]: row["controller"] for row in scenario["switches"]}
        domains = [row["id"] for row in controllers]
    palette = colormaps["tab20"].colors[1::2]
    if len(domains) > len(palette):
        palette = colormaps["tab20"].resampled(len(domains)).colors
    domain_colors = dict(zip(domains, palette))
    placement = {row["placement_switch"]: row["id"] for row in controllers}
    controllable = {node for row in controllers for node in row["controllable_switches"]}
    hosts = {row["id"] for row in scenario["hosts"]}
    graph = nx.Graph()
    graph.add_nodes_from(row["id"] for row in scenario["switches"])
    graph.add_nodes_from(sorted(hosts))
    capacity = {}
    for link in scenario["links"]:
        a, b = link["source"], link["target"]
        graph.add_edge(a, b)
        capacity[a, b] = str(link["capacity_bps"] // 1_000_000)
        if b in hosts:
            owners[b] = owners[a]

    positions = nx.spring_layout(graph, seed=seed, iterations=100)
    size = max(10, 1.1 * len(graph) ** 0.5)
    figure = Figure(figsize=(size, size), layout="constrained")
    axis = figure.subplots()
    font_size = 8 if len(graph) <= 45 else 6
    node_size = 190 if len(graph) <= 45 else 95
    nx.draw_networkx_edges(graph, positions, ax=axis, width=0.8, edge_color="#84909a", alpha=0.7)
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels=capacity, ax=axis, font_size=font_size,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
    )
    roles = [
        ("o", "Switch", [node for node in graph if node not in hosts and node not in placement]),
        ("s", "Host", sorted(hosts)),
        ("D", "Controller location", list(placement)),
    ]
    style = []
    for marker, label, nodes in roles:
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, node_shape=marker, ax=axis,
            node_size=node_size * 1.6 if marker == "D" else node_size,
            node_color=[domain_colors[owners[node]] for node in nodes],
            edgecolors=["#17212b" if node in controllable else "#80909c" for node in nodes],
            linewidths=[2.1 if node in controllable else 0.5 for node in nodes],
        )
        style.append(Line2D([], [], marker=marker, linestyle="none", color="#80909c", label=label))
    if controllable:
        style.append(Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor="#17212b", markeredgewidth=2.1, label="Controllable",
        ))
    style.extend(
        Line2D([], [], marker="o", linestyle="none", color=color, label=domain)
        for domain, color in domain_colors.items()
    )
    labels = {
        node: f"{node}\n{owners[node]}" if pre_merge and node not in hosts
        else f"{node}\n{placement[node]}" if node in placement else node
        for node in graph
    }
    nx.draw_networkx_labels(
        graph, positions, labels, ax=axis, font_size=font_size, font_color="#101820",
    )
    phase = f"\nBefore merging · {len(domains)} Louvain communities" if pre_merge else ""
    axis.set_title(f"{scenario['id']}{phase}\nLink labels in Mbps", pad=10)
    axis.set_axis_off()
    axis.legend(handles=style, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=min(7, len(style)))
    with rc_context({"svg.fonttype": "none", "svg.hashsalt": "eagle"}):
        figure.savefig(path, metadata={"Date": None})
