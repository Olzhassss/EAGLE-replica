import gym,os,sys,time
import networkx as nx
from statistics import mean
import numpy as np
self_healing_path = os.path.dirname(os.getcwd())
sys.path.insert(0, self_healing_path)
from typing import Dict, Tuple
from environment.distributed_env import DistributedEnv

class ParallelEnv:
    # It steps every live agent at once, then waits for all of them to finish and returns the new states, rewards, dones and infos.
    def __init__(self, N) -> None: 
        self._num_agents = N
        self._agents = {}
        self._observation_spaces = {}
        self._action_spaces = {}
        self._reward_spaces = {}
        self._init_flag = False
        self._envs = {}

        for i in range(self._num_agents) : 
            self._agents[i] = '808{}'.format(i)
      

        observation_space = gym.spaces.Dict(
                    {
                        'agent_state': gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(self._num_agents,
                                   4),
                            dtype=np.float32
                        )})
        
        action_space = gym.spaces.Discrete(2)
        reward_space = gym.spaces.Box(low=-5, high=100, shape=(1, ), dtype=np.float32)
       
        for i,agent in enumerate(self._agents): 
            self._observation_spaces[agent] = observation_space
            self._action_spaces[agent] = action_space
            self._reward_spaces[agent] = reward_space

        self.read = 0
        self.portPower = 4.5
        self._lambda = 0.0015

    @property
    def observation_spaces(self) -> gym.spaces.Space:
        return self._observation_spaces

    
    def observation_space(self,agent) -> gym.spaces.Space:
        return self._observation_spaces[agent]

    @property
    def action_spaces(self) -> gym.spaces.Space:
        return self._action_spaces

    
    def action_space(self,agent) -> gym.spaces.Space:
        return self._action_spaces[agent]

    @property
    def reward_spaces(self) -> gym.spaces.Space:
        return self._reward_spaces

    
    def reward_space(self,agent) -> gym.spaces.Space:
        return self._reward_spaces[agent]

    def reset(self) -> np.ndarray:
        obs_n={} 
        if not self._init_flag:
            for i,agent in self._agents.items():
                self._envs[agent] = DistributedEnv(agent,training=True)
            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            for i,agent in self._agents.items(): self._envs[agent].seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            for i,agent in self._agents.items(): self._envs[agent].seed(self._seed)

        for agent, env in self._envs.items():
            obs = env.reset() 
            obs_n[agent] = obs
        self._eval_episode_return = 0. 
        
        return obs_n

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed 
        np.random.seed(self._seed)

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        removing = []
        obs_n,reward_n,done_n,info_n={},{},{},{}
        prev_obs_n = {}
        for i,agent in self._agents.items():
            prev_obs_n[agent] = self._envs[agent].curr_state
            self._envs[agent].parallel_step(actions[i])
            
        time.sleep(4)
        for i,agent in self._agents.items():  
            obs, reward, done, info = self._envs[agent].parallel_info()
            obs_n[agent] = obs
            reward_n[agent] = reward
            done_n[agent] = done
            info_n[agent] = info   
            
        grafo = nx.compose(obs_n['8080'],obs_n['8081']) 
        grafo = nx.compose(grafo,obs_n['8082'])  
        if actions[0] == 0:
            removing.append('s11')
        if actions[1] == 0:
            removing.append('s12')
        if actions[2] == 0:
            removing.append('s13')
        grafo.remove_nodes_from(removing)
      
        return prev_obs_n,obs_n, reward_n, done_n, info_n

    def close(self) -> None:
        self._init_flag = False

    def random_action(self) -> np.ndarray:
        random_action = {}
        for k in range(len(self._action_spaces)) :
            random_action[k] = self._action_spaces[k].sample()
        for k in random_action:
            if isinstance(random_action[k], np.ndarray):
                pass
            
        return random_action

    def hour_carbon(self,graph):
        
        carbon_node = {}
        idx = self.read
        list_nodes = self.carbon_over_time[idx]
        
        for node in ['h1','h2','h3','h4','h5','h6','h7','h8','h9','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13']:
            carbon_node[node] = list_nodes.pop()
        self.read+=1
        return carbon_node
    
    def cost_carbon(self,graph,demands):

        list_P,demands_only = self.get_path_flows(graph,demands)
        carbon_edges = 0
        carbon_per_flow = []
        weight = 1
        edges = graph.edges
        carbon = 0
       
        for d,p in enumerate(list_P):
            for n in p:
                carbon += self.carbon_nodes[n]/len(list_P) * self._lambda * demands_only[d]
        carbon_per_flow.append(carbon*weight)
        carbon_flows=sum(carbon_per_flow)/3600000
        
        for (u,v) in edges:
            carbon_edges+= self.portPower * (self.carbon_nodes[u] + self.carbon_nodes[v] )/2
        
        carbon_edges = carbon_edges/3600000
     
        return carbon_flows+carbon_edges

    def read_node_carbon(self):
        carbon_step = []
        with open("10-17aprile.txt", "r") as file:
            lines_list = file.readlines()
        
        for line in lines_list[0:]:
            carbon_step.append([float(val) for val in line.split()])
        
        return carbon_step

    def get_path_flows(self,graph,demands):
        demands1 = demands['8080']
        demands2 =demands['8081'] 
        demands3 =demands['8082'] 
        demands1.update(demands2) 
        demands1.update(demands3)
        demands = {x:y for x,y in demands1.items() if y!=0}
        demands_only = []
        
        paths_tot = []
        for k,v in demands.items():
            src,dst = k.split(' ')
            
            paths = [p for p in nx.all_shortest_paths(graph,src,dst)]
            if len(paths) > 1:
                for p in paths:
                    if self.p_has_supp(p):
                        choose = p
                        break
            else:
                choose = paths[0]
          
            paths_tot.append(choose)
            demands_only.append(v/1024/1024)
      
        return paths_tot,demands_only

    def p_has_supp(self,path):
        for supp in ['s11','s12','s13']:
            if supp in path:
                return True
        return False
