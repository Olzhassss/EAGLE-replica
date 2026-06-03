#! /usr/bin/env python
from benchmark_consts import get_args
import sys
sys.path.append('..')
from problem import Problem
from balanced_louvain import bpartitions
from vis import vis_partitions

def benchmark(topo):
    
    subnets = []

    graph_name, topo_fname = topo
    G = get_graph(topo_fname)
    copy_graph = G.copy()
        
    part = bpartitions(copy_graph,resolution=0.1,gamma=60,epsilon=10,rho=0.9,seed=42)
    
    *_, last = part 
    for _set in last:
        net = copy_graph.subgraph(_set).copy()
        subnets.append(net)
    
    vis_partitions(copy_graph,last,graph_name)
    return


def get_graph(topo_fname):

    G = Problem.from_file(topo_fname)
    return G

                
if __name__ == '__main__':
    
    args, topo = get_args()
    benchmark(topo)
