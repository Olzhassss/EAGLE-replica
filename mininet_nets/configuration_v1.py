from mininet_nets.manageInterfaces import Interfaces
import networkx as nx
import json,crc32c

class Configuration_V1:

    def __init__(self):
        self.datapaths = {} 
        self.switches={} 
        self.supp_sw=None 
        self.enabled = {11: 1, 12: 1, 13: 1}
        self.topology_change=0
        self.hosts={} 
        self.macs={} 
        self.global_net=nx.Graph()
        self.removed_edges=[]
        self.topology=self.load_topo('topology.json')

    def load_topo(self,json_path):

        with open(json_path, 'r') as f:
            graph_dict = json.load(f)
            graph = nx.node_link_graph(graph_dict)
            return nx.Graph(graph)

    def create_net(self):
        
        self.nodes=list(self.switches.values())+list(self.macs.keys())
        self.net=self.topology.subgraph(self.nodes).copy()
        self.global_net=self.net   
        
        for u,v,data in self.topology.edges.data():
            if (u in self.switches.values() and v not in self.switches.values()) or (v in self.switches.values() and u not in self.switches.values()):
                if 'port1' in data.keys() and 'port2' in data.keys(): 
                    self.removed_edges.append((u,v,data)) 
                         
        self.graph = self.net.copy()
        self.graph.add_edges_from(self.removed_edges)

    def get_hosts_connected_to_switch(self,sw_id):

        sw='s{}'.format(sw_id)
        self.switches[sw_id]=sw 
        topo=self.topology
        nodes=dict(topo.nodes(data=True))
        for n in topo.neighbors(sw):
            node=nodes[n]
            if node.get('isHost'):
                if sw_id not in self.hosts.keys():
                    self.hosts[sw_id]=[]
                self.hosts[sw_id].append(n) 
                self.macs[n]=node.get('mac')

    def config_switch(self, datapath, default=False):

        print(f'new datapath id: {datapath.id}')
        switch_id = datapath.id 
        sw='s{}'.format(switch_id)
        self.datapaths[switch_id] = datapath
        self.remove_flows(datapath)
        self.get_hosts_connected_to_switch(switch_id)
         
        if switch_id == 11 or switch_id == 12 or switch_id == 13:
            self.supp_sw=switch_id
            self.edges=[(u,v,d) for u,v,d in self.topology.edges.data() if u==sw or v==sw]

    def table_clear(self):
        for dp in self.datapaths.values():
            self.remove_flows(dp)
    

    @staticmethod
    def remove_flows(datapath):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch()
        instructions = []
        Interfaces.remove_flows(datapath,match,instructions) 
         

    def act(self, action):
        ''' Take the action from the Environment and enable or disable the supporting switch accordingly '''
        action=action[0]

        if action == 1:
            self.enable_supp_sw(self.supp_sw)
        else:
            self.disable_supp_sw(self.supp_sw)     


    def enable_supp_sw(self,sw_id):
        ''' Enable the supporting switch and add it to the virtual network topology '''
        if self.enabled[sw_id] == 1:
            return
        
        sw=self.switches[sw_id]
        self.net.add_node(sw)
        self.net.add_edges_from(self.edges)
        self.global_net.add_node(sw)
        self.global_net.add_edges_from(self.edges)
        self.enabled[sw_id]=1
        self.topology_change=1
        self.table_clear()
        self.route()

    def disable_supp_sw(self,sw_id):
        ''' Disable the supporting switch and remove it from the virtual network topology '''
        if self.enabled[sw_id] == 0:
            return
        
        sw=self.switches[sw_id]
        self.net.remove_node(sw)
        self.global_net.remove_node(sw)
        self.enabled[sw_id]=0
        self.topology_change=1
        self.table_clear()
        self.route()


    def route(self):  
        ''' Apply ECMP routing rules to the switches based on the current virtual network topology '''
        for dpid in self.datapaths.keys():
            sw_name=self.switches[dpid] 
            if dpid == self.supp_sw and self.enabled[self.supp_sw] == 0:
                continue 
            for sw_dst in [n for n,d in self.global_net.nodes(data=True) if d.get('isSwitch')]: 
                if sw_dst == sw_name: 
                    if dpid in self.hosts.keys():
                        for host in self.hosts[dpid]:
                            out_port=self.global_net[sw_name][host]['port1']
                            for src in self.macs.keys(): 
                                if src != host: 
                                    self.add_l2_flow(self.datapaths[dpid], self.macs[src], self.macs[host], out_port)    
  
                else: 
                    hosts_dst=[]
                    flag = 0

                    try:
                        paths = [p for p in nx.all_shortest_paths(self.global_net,sw_name,sw_dst)]
                    except nx.exception.NetworkXNoPath:
                        continue

                    for n in self.global_net.neighbors(sw_dst):
                        
                        if n in [h for h,d in self.global_net.nodes(data=True) if d.get('isHost')]:
                            hosts_dst.append(n)
                    
                    for host in hosts_dst:
                        if len(paths) == 1:
                            next_hop = paths[0][1] 
                            link=self.global_net[sw_name][next_hop]
                            
                            if link['node1'] == sw_name:
                                out_port=link['port1']
                            else:
                                out_port=link['port2'] 
                            
                            for src in self.macs.keys():
                                if src != host: 
                                    self.add_l2_flow(self.datapaths[dpid], self.macs[src], self.macs[host], out_port)       
                        else: 
                            num_p=len(paths)
                            for path in paths:
                                if self.switches[self.supp_sw] in path:
                                    next_hop = path[1] 
                                    link=self.global_net[sw_name][next_hop]
                            
                                    if link['node1'] == sw_name:
                                        out_port=link['port1']
                                    else:
                                        out_port=link['port2'] 
                            
                                    for src in self.macs.keys(): 
                                        if src != host: 
                                            self.add_l2_flow(self.datapaths[dpid], self.macs[src], self.macs[host], out_port) 
                                            flag=1
                            if flag==0:
                                for src in self.macs.keys(): 
                                    if src != host:
                                        session = src+host+self.macs[src]+self.macs[host]
                                        path_index = crc32c.crc32c(session.encode()) % num_p 
                                        next_hop = paths[path_index][1] 
                                        link = self.global_net[sw_name][next_hop]
                            
                                        if link['node1'] == sw_name:
                                            out_port=link['port1']
                                        else:
                                            out_port=link['port2']

                                        self.add_l2_flow(self.datapaths[dpid], self.macs[src], self.macs[host], out_port)       
                                    

    @staticmethod
    def add_l2_flow(datapath, mac_src, mac_dst, out_port):
        ''' Add a flow to the switch to forward packets with source mac_src and destination mac_dst to the output port out_port '''
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(out_port)]

        match = parser.OFPMatch(eth_src=mac_src,eth_dst=mac_dst)
        Interfaces.add_flow(datapath, 1, match, actions)

    def normalize_state(self,state): 
        ''' Normalize the state representation '''
        new_state = {}
        new_state['data'] = {}
        
        if len(state['data'].keys()) == 0: return

        for sw_id,sw in self.switches.items():

            new_state['data'][sw] = {} 

            for port in state['data'][sw_id]['out_packets'].keys():
                neigh = port_to_node(self.graph,sw,port)
                bw = node_to_node_interface_bw(self.graph,sw,neigh)
                ratio = state['data'][sw_id]['out_packets'][port]*8/1024/1024/bw
                
                if ratio > 1: ratio = 1
                if ratio < 0.01: ratio = 0
                new_state['data'][sw][neigh]=ratio
                

            for port in state['data'][sw_id]['in_packets'].keys():
                neigh = port_to_node(self.graph,sw,port)
                bw = node_to_node_interface_bw(self.graph,sw,neigh)
                ratio = state['data'][sw_id]['in_packets'][port]*8/1024/1024/bw
                
                if ratio > 1: ratio = 1
                if ratio < 0.01: ratio = 0 
                if neigh not in new_state['data']: new_state['data'][neigh]={}
                new_state['data'][neigh][sw]=ratio
        
        return new_state


def port_to_node(topo,sw,port): 
    for node in topo.neighbors(sw):
        link = topo[sw][node]
        if link['node1'] == sw:
            p = link['port1']
        else:  
            p = link['port2'] 
        if port == p:
            return node

def node_to_node_interface_bw(topo,sw,neigh): 
    return topo[sw][neigh]['bw']
