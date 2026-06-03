import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def vis_partitions(copy_graph, last,problem_name):
    
    color_cycle = plt.cm.rainbow(np.linspace(0, 1, len(last)))
    node_groups = {}
    for group, color in zip(last, color_cycle):
        for node in group:
            node_groups[node] = color
    colors = [node_groups.get(node, 'black') for node in copy_graph.nodes()]
    pos = nx.spring_layout(copy_graph) 
    plt.figure(figsize=(15, 15))
    nx.draw(copy_graph, pos,node_color=colors, with_labels=True, node_size=500, font_color='white')
    plt.savefig(f'{problem_name}_subnets.png')