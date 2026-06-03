import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
import json, nnpy
from operator import attrgetter
import crc32c,time
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.ofproto import ofproto_v1_3
from ryu.services.protocols.ovsdb import event as ovsdb_event
import networkx as nx
import datetime

from webob import Response

from config_reader import Reader
from mininet_nets.configuration_v1 import Configuration_V1

simple_switch_instance_name = 'simple_switch_api_app'
url_stats = '/network/stats'
url_switches = '/network/list_switches'
url_actions = '/model/actions'


class SimpleMonitor13(app_manager.RyuApp):
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.system_id = 0
        self.prev = {} 
        self.prev_flows={} 
        self.last_timestamp = {}
        self.state={} 
        self.stats={} 
        self.state['type'] = 'information'
        self.state['data'] = {}
        self.flows={} 
        self.throughput={} 
        self.demands = {}
        self.config_file = 'config.ini'
        self.config = Reader(self.config_file)
        self.time_interval = self.config.frequency_request() 
        self.node = None
        self.setup = Configuration_V1()
        wsgi = kwargs['wsgi']
        wsgi.register(SimpleSwitchController,
                      {simple_switch_instance_name: self})
       
        hub.spawn(self._monitor)
        hub.spawn(self._listen)


    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                if datapath.id == 12: 
                    self.node = '127.0.0.1:8000'
                    self.cls=['127.0.0.1:8001','127.0.0.1:8002']
                if datapath.id == 11:
                    self.node = '127.0.0.1:8001'
                    self.cls=['127.0.0.1:8000','127.0.0.1:8002']
                if datapath.id == 13:
                    self.node = '127.0.0.1:8002'
                    self.cls=['127.0.0.1:8001','127.0.0.1:8000']
                
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath

                self.setup.config_switch(datapath)
            
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    @set_ev_cls(ovsdb_event.EventNewOVSDBConnection)
    def handle_new_ovsdb_connection(self, ev):
        self.system_id = ev.system_id
        address = ev.client.address
        self.logger.info('New OVSDB connection from system-id=%s, address=%s', self.system_id, address)

    def _listen(self):
        ''' Listen to other controllers and update the global topology '''
        hub.sleep(1)

        while self.node == None:
            hub.sleep(1)

        self.sub = nnpy.Socket(nnpy.AF_SP, nnpy.REP)
        self.sub.bind('tcp://{}'.format(self.node))

        self.sub_c1= nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
        self.sub_c1.connect('tcp://{}'.format(self.cls[0])) 

        self.sub_c2= nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
        self.sub_c2.connect('tcp://{}'.format(self.cls[1]))

        self.graph_c1=None
        self.graph_c2=None
        
        self.setup.create_net() 
        
         
        msg={'sender':self.node, 'graph':nx.node_link_data(self.setup.net), 'edges':self.setup.removed_edges}
        
        self.sub_c1.send(json.dumps(msg))
        self.sub_c2.send(json.dumps(msg))

        while True: 
            try :
                msg = self.sub.recv(1)
                msg = json.loads(msg) 
                sender=msg['sender']
                subgraph=msg['graph']
                edges=msg['edges']
                subgraph=nx.node_link_graph(subgraph)
                if sender == self.cls[0]:
                    self.graph_c1=subgraph
                    self.edges_c1=edges
                else :
                    self.graph_c2=subgraph
                    self.edges_c2=edges
                self.create_topo()
            except:
                hub.sleep(0.2)
 
    def create_topo(self):

        if self.graph_c1 is not None and self.graph_c2 is not None:
            self.setup.global_net=nx.compose(self.setup.net,self.graph_c1)
            self.setup.global_net=nx.compose(self.setup.global_net,self.graph_c2)
            
            self.setup.global_net.add_edges_from(self.edges_c1)
            self.setup.global_net.add_edges_from(self.edges_c2) 
            
            for edge in self.setup.global_net.edges:
                node1=edge[0]
                node2=edge[1]
                nodes = self.setup.global_net.nodes
                if nodes[node1].get('isHost'):
                    self.setup.macs[node1]=nodes[node1].get('mac')
                if nodes[node2].get('isHost'):
                    self.setup.macs[node2]=nodes[node2].get('mac')

            self.setup.table_clear() 
            self.setup.route() 

    def topology_change(self):

        ''' Notify other controllers about the topology change '''

        if self.setup.topology_change == 1:
            msg={'sender' : self.node, 'graph' : nx.node_link_data(self.setup.net), 'edges' : self.setup.removed_edges}
            self.sub_c1.send(json.dumps(msg))
            self.sub_c2.send(json.dumps(msg))
            self.setup.topology_change = 0

    def _monitor(self):
        hub.sleep(1)
        
        while True:
            
            for dp in self.datapaths.values():
                self._request_stats(dp)
                if dp.id in self.setup.hosts.keys(): 
                    self._request_flows(dp)
            
            self.flows=self.throughput 
            self.stats=self.setup.normalize_state(self.state)
        
            hub.sleep(self.time_interval) 


    def _request_stats(self, datapath):
        """
        Request statistics to the switch
        :param datapath: switch identifier
        """
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def _request_flows(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        cookie = cookie_mask = 0
        req_th = parser.OFPFlowStatsRequest(datapath, 0,
                                         ofproto.OFPTT_ALL,
                                         ofproto.OFPP_ANY, ofproto.OFPG_ANY,
                                         cookie, cookie_mask)
        datapath.send_msg(req_th)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """
        Handle switch response
        :param ev: response
        """
        body = ev.msg.body
        datapath_id = ev.msg.datapath.id
        self.packets_on_each_port(body, datapath_id)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        datapath_id = ev.msg.datapath.id
        self.flow_stats(body,datapath_id)

    def flow_stats(self,body,datapath_id):
        ''' Calculate the throughput and demands for each flow '''
        for stat in body:
            if 'eth_src' not in stat.match or 'eth_dst' not in stat.match: continue
            src=stat.match['eth_src']
            dst=stat.match['eth_dst']
            host_src=[k for k, v in self.setup.macs.items() if v == src][0]
            host_dst=[k for k, v in self.setup.macs.items() if v == dst][0]

            if host_src in self.setup.hosts[datapath_id] or host_dst in self.setup.hosts[datapath_id]:
                session=src+dst+str(src)+str(dst)
                hash_key = crc32c.crc32c(session.encode())
                kstr = host_src+' '+host_dst
                total=stat.byte_count
 
                if hash_key not in self.prev_flows.keys(): 
                    self.prev_flows[hash_key] = total
                else: 
                    if total >= self.prev_flows[hash_key]:
                        self.throughput[hash_key]=total - self.prev_flows[hash_key]
                        self.demands[kstr] = total - self.prev_flows[hash_key]
                    else:
                        self.throughput[hash_key]=total 
                        self.demands[kstr] = total
                self.prev_flows[hash_key]=total 
                

    def packets_on_each_port(self, body, datapath_id):
        
        in_packets = {}
        out_packets = {}

        for stat in sorted(body, key=attrgetter('port_no')):
            if stat.port_no > ofproto_v1_3.OFPP_MAX:
                continue
            
            in_packets[stat.port_no] = stat.rx_bytes
            out_packets[stat.port_no] = stat.tx_bytes

        if datapath_id not in self.prev:
            self.logger.info(f'First response from switch {datapath_id}')
            self.last_timestamp[datapath_id] = datetime.datetime.now()
            self.prev[datapath_id] = {}
        else:
            self.last_timestamp[datapath_id] = datetime.datetime.now()
            in_diff = {}
            out_diff = {}
            
            for port, prev_in_packets in self.prev[datapath_id]['in_packets'].items():
                in_diff[port] = in_packets[port] - prev_in_packets
                
            for port, prev_out_packets in self.prev[datapath_id]['out_packets'].items():
                out_diff[port] = out_packets[port] - prev_out_packets
                
            self.state['data'][datapath_id] = {
                'timestamp': str(self.last_timestamp[datapath_id]),
                'in_packets': in_diff,
                'out_packets': out_diff
            }

        self.prev[datapath_id]['in_packets'] = in_packets
        self.prev[datapath_id]['out_packets'] = out_packets


class SimpleSwitchController(ControllerBase):
    ''' Controller for the REST API to interact with the switch and get the statistics asked by the Environment '''
    def __init__(self, req, link, data, **config):
        super(SimpleSwitchController, self).__init__(req, link, data, **config)
        self.simple_switch_app = data[simple_switch_instance_name]

    @route('simpleswitch', url_stats, methods=['GET'])
    def get_stats(self, req, **kwargs):
        data = {}
        simple_switch = self.simple_switch_app
        if len(simple_switch.stats['data'].keys()) == 0: return Response(status=400)
        data['stats'] = simple_switch.stats
        data['throughput'] = simple_switch.flows
        data['demands'] = simple_switch.demands
        data['topology'] = nx.node_link_data(simple_switch.setup.graph) 
        body = json.dumps(data)
        return Response(content_type='application/json', body=body, charset='utf-8')


    @route('simpleswitch', url_actions, methods=['PUT'])
    def put_actions(self, req, **kwargs):
        simple_switch = self.simple_switch_app

        try:
            new_action = req.json if req.body else {}
            if len(new_action) != 1:
                raise ValueError
            for value in new_action:
                if value != 0 and value != 1:
                    raise ValueError
            start_time=time.time()
            simple_switch.setup.act(new_action)
            print('Action processing time:', time.time() - start_time) # includes changing routing
            simple_switch.topology_change()

        except ValueError:
            return Response(status=400)

        body = json.dumps(new_action)
        return Response(content_type='application/json', body=body, charset='utf-8', status=200)
