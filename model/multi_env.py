import os, sys
import numpy as np
import networkx as nx
from karateclub import FeatherGraph
self_healing_path = os.path.dirname(os.getcwd())
sys.path.insert(0, self_healing_path)
from ding.envs import BaseEnvTimestep, BaseEnv
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from typing import List, Dict
import gym
from parallel_env_net import ParallelEnv

@ENV_REGISTRY.register("multi_network")
class NetworkEnv(BaseEnv):
    
    def __init__(self, cfg: dict) -> None: 
        self._cfg = cfg
        self._num_agents = self._cfg.env.n_agent
        self._init_flag = False
        self._max_steps = self._cfg.env.max_steps
        self._step_count = 0
        self.n =0
        self.mean = np.zeros(8)
        self.mean_diff = np.zeros(8)
        self.var= np.ones(8)
        self.minimum = []
        self.maximum = []

    def reset(self) -> np.ndarray:

        if not self._init_flag:
            self._env = ParallelEnv(N=self._num_agents)
            self._agents = self._env._agents
            self.action_space = gym.spaces.Dict({agent: self._env.action_space(agent) for agent in self._agents.keys()})
            self.observation_space = gym.spaces.Dict({agent: self._env.observation_space(agent) for agent in self._agents.keys()})
            self.reward_space = gym.spaces.Dict({agent: self._env.reward_space(agent) for agent in self._agents.keys()})
            
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            
        obs = self._env.reset() 
        obs_n = self._process_obs(obs)
        
        self._eval_episode_return = 0.
        self._step_count = 0 

        return obs_n

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed 
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        done = 0
        self._step_count += 1

        assert isinstance(action, list), type(action)

        processed_action = self._process_action(action)
        
        prev_obs, obs, rew, done, info = self._env.step(processed_action)
        prev_obs_n = self._process_obs(prev_obs) 
        obs_n = self._process_obs(obs)
        rew_n = [r for i,r in rew.items()]
        
        self._eval_episode_return += np.mean(rew_n) 

        if self._step_count >= self._max_steps:
            done = 1
            self._step_count = 0
        else:
            done = 0
        if done: 
            info['eval_episode_return'] = self._eval_episode_return
            
            print("Return of the episode:",self._eval_episode_return) 
        info['action'] = action
        
        return BaseEnvTimestep(obs_n, rew_n, done, info)


    def close(self) -> None:
        self._init_flag = False

    def _process_action(self, action: List) -> Dict[str, np.ndarray]:  

        dict_action = {}
        for i, agent in enumerate(self._agents):
            agent_action = action[i]
            if agent_action.shape == (1, ):
                agent_action = agent_action.squeeze()  
            dict_action[agent] = int(agent_action)
        return dict_action

    def _process_obs(self, obs: 'torch.Tensor') -> np.ndarray: 

        ''' Process the observation from the environment and return the state representation for the agents. 
            Each observation is a graph, and we use the FeatherGraph algorithm to extract the features 
            of the graph and concatenate them to get the state representation. '''
        
        state = {}
        agents_embeddings = {}
        global_graph = nx.Graph()
        for i in range(self._num_agents):
            global_graph = nx.compose(global_graph,obs[self._agents[i]])
        
        for i in range(self._num_agents):
            agents_embeddings[self._agents[i]]=self.agent_FG(obs[self._agents[i]])

        agents_embeddings=dict(sorted(agents_embeddings.items()))
        
        embeddings = np.concatenate([agents_embeddings['8080'], agents_embeddings['8081'], agents_embeddings['8082']],1)
        
        embeddings = self.normalize(embeddings)
        
        state['global_obs'] =  embeddings

        return state

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        for k in random_action:
            if isinstance(random_action[k], np.ndarray):
                pass
            elif isinstance(random_action[k], int):
                random_action[k] = to_ndarray([random_action[k]], dtype=np.int64)
        return random_action

    def __repr__(self) -> str:
        return "Network Env({})".format(self._cfg.env_id)

    def normalize(self,x):
        x=x[0]
         
        self.minimum = np.array([2705., 2706.,   41. ,   -1., 2751., 2751.,   42. ,   -1. ,2705., 2706.,   41.  ,  -1.])
        self.maximum = np.array([4969., 4970.,   82.,    1., 4542., 4543.,   75.,    1., 4918., 4919. ,  81.  ,  1.])

        old_range = self.maximum-self.minimum
        new_range = 100
        x_norm = ((x-self.minimum)/old_range)*new_range
       
        for i in range(len(x_norm)):
            if np.isnan(x_norm[i]): x_norm[i]=0
            if np.isinf(x_norm[i]): x_norm[i]=0 

        return x_norm

    def agent_FG(self,graph):

        graph = nx.convert_node_labels_to_integers(graph)
        model =  FeatherGraph(order=1,eval_points=1) 

        model.fit([graph])
        X = model.get_embedding()
        
        for i in range(len(X[0])):
            X[0][i]=round(X[0][i],4)*10000

        return X


    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]
