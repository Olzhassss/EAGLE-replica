import configparser,json,os,time
from json import JSONEncoder
import networkx as nx,os
import numpy as np
import requests,random


class CommunicationError(Exception):
    """Exception raised for error in communication with Ryu controller"""
    pass


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def _read_config_file(config_file):

    current_path = os.path.dirname(__file__)
    config_path = os.path.join(current_path, config_file)
    print(f'Reading config file {config_path}')
    config = configparser.ConfigParser()
    config.read(config_path)
    num_intf = int(config.get('DEFAULT', 'NumIntf'))
    print(f'Total number of interfaces: {num_intf}')
    supporting_switches = int(config.get('DEFAULT', 'NumSuppSwitch'))
    print(f'Total number of supporting switches: {supporting_switches}')
    op_penalty = int(config.get('DEFAULT', 'OPPenalty'))
    print(f'Overprovisioning penalty: {op_penalty}')
    helped_switches = json.loads(config.get('DEFAULT', 'HelpedSwitches'))
    print(f'Switch indexes of helped switches: {helped_switches}')
    frequency = json.loads(config.get('DEFAULT', 'FrequencyData'))
    print(f'Action every {frequency} secs')

    return num_intf, supporting_switches, op_penalty, helped_switches, frequency


class DistributedEnv:
    ''' Environment for the decentralized RL agent to interact with the Ryu controller and get the network statistics, and give the actions to enable or disable the supporting switch accordingly '''
    def __init__(self, port, training=True):
        config_file = 'config_{}.ini'.format(port)
        num_intf, supporting_switches, op_penalty, helped_switches, frequency = _read_config_file(config_file)

        self.state_size = num_intf
        self.supporting_switches = supporting_switches
        self.action_size = 2 ** supporting_switches
        self.op_penalty = op_penalty
        self.helped_switches = helped_switches
        self.frequency = frequency
        self.throughputs = []
        self.curr_state = None
        self.curr_action = None
        self.port=port
        self.training=training
        self.alpha=1/100
        self.tp_norm=40
        if self.training == False: self.op_penalty = 15

        self.ip_controller = 'http://127.0.0.1:{}'.format(port)
        self.url_stats = '/network/stats'
        self.url_switches = '/network/list_switches'
        self.url_actions = '/model/actions'

        self.n = np.zeros(self.state_size)
        self.mean = np.zeros(self.state_size)
        self.mean_diff = np.zeros(self.state_size)
        self.var = np.ones(self.state_size)
        
        self.actions={
            0: [0],
            1: [1]
        } 

    def reset(self):
        _,state=self._get_state()
        return state

    def seed(self,seed):
        random.seed(seed) 

    def _encode_action(self, int_action):
        bin(int_action).replace('0b', '')

    def parallel_step(self, action_id):

        action = self.actions[action_id]
        self._put_action(action)

    def parallel_info(self):

        demands,state = self._get_state()
        reward = self._get_reward()
        done = False
        info = demands
        
        return state, reward, done, info

    def step(self, action_id):
        ''' Take the action from the agent, give it to the controller, wait for the next state and reward, and return them to the agent '''
        action = self.actions[action_id]
        self._put_action(action)
        time.sleep(self.frequency) 
        demands,state = self._get_state()
        reward = self._get_reward()
        done = False
        info = demands
        
        return state, reward, done, info

    def _put_action(self, action):
        '''
            Give the action to the controller
        '''
        try:
            url = self.ip_controller + self.url_actions
            data = json.dumps(action)
           
            response = requests.put(url, data)
            if response.status_code != 200:
                raise CommunicationError
            self.curr_action = action
        except TypeError or CommunicationError:
            print(f"Error. Action = {action}")

    def _get_state(self):
        '''
           Ask to the controller the stats
           :return: state
        '''
        throughputs=[]
        url = self.ip_controller + self.url_stats
        response = requests.get(url)
        # For successful API call, response code will be 200 (OK)
        if not response.ok:
            raise CommunicationError

        jData = json.loads(response.content)
        link_ratio = jData['stats']['data'] 
           
        flows = jData['throughput'] 
        demands = jData['demands'] 
        
        graph = nx.node_link_graph(jData['topology'])
        
        graph = self.mapping_info(graph,link_ratio) 
        state = graph 
        
        for v in flows.values():
            if v != 0:
                throughputs.append(v)
        
        self.throughputs = throughputs
        self.curr_state = state
       
        return demands,state

    def _get_reward(self):
        ''' Calculate the reward based on the throughputs and the overprovisioning penalty '''
        action = self.curr_action
        sum_tp = 0
        overprovisioning = 0
        tp=0
        
        for i in self.throughputs:
            sum_tp += i
        
        if len(self.throughputs) > 0:
            tp = sum_tp / len(self.throughputs) *8/1024/1024 
        
        for enabled_switch in action:
            overprovisioning += enabled_switch * self.op_penalty
        
        reward = tp/self.tp_norm - overprovisioning*self.alpha
        return int(reward*100)

    def mapping_info(self,graph,info_tuple):
        
        ''' Map the link ratio information to the graph edges as weights '''

        for sw in info_tuple:
            for neigh in info_tuple[sw].keys():
                ratio=info_tuple[sw][neigh]
                if 'weight' in graph[sw][neigh]:
                    prev = graph[sw][neigh]['weight']
                    if ratio > prev:
                        graph[sw][neigh]['weight']=ratio
                    
                else:
                    graph[sw][neigh]['weight']=ratio
                  
        return graph

    

