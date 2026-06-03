from collections import defaultdict
from networkx.utils import py_random_state
import networkx as nx

@py_random_state("seed")
def bpartitions(
    G, weight="weight", resolution=1, seed=None, gamma=10, epsilon=2, rho=0
):
    '''Balanced Louvain partitioning algorithm, with a penalty rho for unbalanced partitions. 
    The balance is defined by the difference between the sizes of the sets, which should be less than epsilon. 
    The capacity is the maximum size of a set, which should be less than gamma.'''
    
    partition = [{u} for u in G.nodes()]
    
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _first_level(
        graph, m, partition, gamma, epsilon, rho, resolution, is_directed,seed
    )
    
    improvement = True
    while improvement:
        
        yield [s.copy() for s in partition]
        
        graph = gen_graph(graph, inner_partition)
       
        partition, inner_partition, improvement = _first_level(
            graph, m, partition, gamma, epsilon, rho, resolution, is_directed,seed
        )       
   

def _first_level(G, m, partition, gamma, epsilon, rho, resolution=1, is_directed=False, seed=0):

    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
        in_degrees = dict(G.in_degree(weight="weight"))
        out_degrees = dict(G.out_degree(weight="weight"))
        Stot_in = list(in_degrees.values())
        Stot_out = list(out_degrees.values())
        
        nbrs = {}
        for u in G:
            nbrs[u] = defaultdict(float)
            for _, n, wt in G.out_edges(u, data="weight"):
                nbrs[u][n] += wt
            for n, _, wt in G.in_edges(u, data="weight"):
                nbrs[u][n] += wt
    else:
        degrees = dict(G.degree(weight="weight"))
        Stot = list(degrees.values())
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
        
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            
            best_mod = 0
            best_com = node2com[u]
            weights2com = neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree = in_degrees[u]
                out_degree = out_degrees[u]
                Stot_in[best_com] -= in_degree
                Stot_out[best_com] -= out_degree
                remove_cost = (
                    -weights2com[best_com] / m
                    + resolution
                    * (out_degree * Stot_in[best_com] + in_degree * Stot_out[best_com])
                    / m**2
                )
            else:
                degree = degrees[u]
                Stot[best_com] -= degree
                remove_cost = -weights2com[best_com] / m + resolution * (
                    Stot[best_com] * degree
                ) / (2 * m**2)
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution
                        * (
                            out_degree * Stot_in[nbr_com]
                            + in_degree * Stot_out[nbr_com]
                        )
                        / m**2
                    )
                else:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution * (Stot[nbr_com] * degree) / (2 * m**2)
                    )
                if nbr_com != node2com[u]:
                    gain = gain + rho
                    origin_set_size = len(partition[node2com[u]])-1
                    dst_set_size = 1 + len(partition[nbr_com])
                    old_size = [len(p) for p in partition]
                    old_size[nbr_com] = dst_set_size
                    old_size[node2com[u]] = origin_set_size

                    balance = is_balanced(old_size,epsilon)
                    if not balance:
                        gain = gain - rho
                
                if gain >= best_mod:
                    best_mod = gain
                    best_com = nbr_com

            if is_directed:
                Stot_in[best_com] += in_degree
                Stot_out[best_com] += out_degree
            else:
                Stot[best_com] += degree

            if best_com != node2com[u]:
                if len(partition[best_com]) + len(partition[node2com[u]]) > gamma:
                    continue  
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com

    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement

def is_balanced(sizes,balance):
    ''' Check if the partition is balanced, i.e. the difference between the sizes of the sets is less than epsilon. '''
    n = len(sizes)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(sizes[i] - sizes[j])
            if diff>balance: return False 

    return True

def neighbor_weights(nbrs, node2com):
    """Calculate weights between node and its neighbor communities.

    Parameters
    ----------
    nbrs : dictionary
           Dictionary with nodes' neighbours as keys and their edge weight as value.
    node2com : dictionary
           Dictionary with all graph's nodes as keys and their community index as value.

    """
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, weight=wt + temp)
    return H


def convert_multigraph(G, weight, is_directed):
    """Convert a Multigraph to normal Graph"""
    
    H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H
