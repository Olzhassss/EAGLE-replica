import re
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from numbers import Real


class Problem(object):
    def __init__(self,
                 G,
                 seed=0,
                 **kwargs):
        self.G = G
        self.capacity_seed = seed


    ############################
    # Public class methods:    #
    # Instantiate new Problems #
    ############################
    @classmethod
    def from_file(cls, topology_fname, old_way=True):
        if topology_fname.endswith('-edgelist.txt'):
            G = nx.read_edgelist(topology_fname,
                                 create_using=nx.Graph,
                                 nodetype=int,
                                 data=[('capacity', float),('traffic',float)])
        elif topology_fname.endswith('.json'):
            G = Problem._read_graph_json(topology_fname)
            G.remove_nodes_from(['h1','h2','h3','h4','h5','h6','h7','h8','h9'])
          
        elif topology_fname.endswith('.graphml'):
            G = Problem._read_graph_graphml(topology_fname)
        elif topology_fname.endswith('.dot'):
            G = Problem._read_graph_dot(topology_fname, old_way=old_way)

        return G

    def copy(self):
        G = self.G.copy()
        tm = self.traffic_matrix.copy()
        problem = Problem(G, tm)
        problem.name = self.name
        tm._problem = problem
        problem.capacity_seed = self.capacity_seed
        return problem

    ##############
    # Properties #
    ##############
    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, G):
        self._G = G

    @property
    def edges_list(self):
        if not hasattr(self, '_edges_list'):
            self._edges_list = list(self.G.edges)
        return self._edges_list


    @property
    def edge_idx(self):
        return {edge: e for e, edge in enumerate(self.G.edges)}


    ##########################
    # Private static methods #
    ##########################
    @staticmethod
    def _read_graph_json(fname):
        assert fname.endswith('.json')
        with open(fname) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

    @staticmethod
    def _read_graph_graphml(fname):
        assert fname.endswith('.graphml')
        file_G = nx.read_graphml(fname).to_directed()
        if isinstance(file_G, nx.MultiDiGraph):
            file_G = nx.Graph(file_G)

        G = []
        # Pick largest strongly connected component
        for scc_ids in nx.strongly_connected_components(file_G):
            scc = file_G.subgraph(scc_ids)
            if len(scc) > len(G):
                G = scc
        G = nx.convert_node_labels_to_integers(G)
        G = G.to_undirected()
        # For TZ topologies, assume every link has 1000 Mbps of capacity
        for u, v in G.edges():
            G[u][v]['capacity'] = 1000.0
            G[u][v]['traffic'] = 0
        return G
