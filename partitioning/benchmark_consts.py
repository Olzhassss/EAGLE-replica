from collections import defaultdict
from glob import iglob

import argparse
import os
import sys
sys.path.append('..')

GRAPHS = []
GRAPHS_NAMES = [
    'GtsCe.graphml', 'UsCarrier.graphml', 'Cogentco.graphml', 'Colt.graphml',
    'TataNld.graphml', 'Deltacom.graphml', 'DialtelecomCz.graphml',
    'Uninett2010.graphml', 'Interoute.graphml', 'Ion.graphml', 'Kdl.graphml'
]

for graph_name in GRAPHS_NAMES:
    topo_fname = os.path.join('topology-zoo',
                                  graph_name)
    GRAPHS.append((graph_name,topo_fname))
    

def get_topo(args):

    for i,graph_name in enumerate(GRAPHS_NAMES):
        if args.topo+'.graphml' == graph_name:
            return GRAPHS[i]
        if args.topo+'.json' == graph_name:
            return GRAPHS[i]
    return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo',
                        type=str,
                        required=False,
                        default='GtsCe')
    parser.add_argument('--capacity',
                        type=int,
                        required=False,
                        default=60,
                        help='Maximum capacity of a community')
    parser.add_argument('--difference',
                        type=int,
                        required=False,
                        default=10,
                        help='Maximum allowed difference between community sizes')
    parser.add_argument('--rho',
                        type=float,
                        required=False,
                        default=0.9,
                        help='Penalty parameter for unbalanced communities')
    
    args = parser.parse_args()
    return args, get_topo(args)
