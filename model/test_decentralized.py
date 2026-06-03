import os, sys
self_healing_path = os.path.dirname(os.getcwd())
sys.path.insert(0, self_healing_path)
import torch 
from statistics import mean
from easydict import EasyDict 
from ding.config import compile_config
from ding.policy import DecentralizedPolicy
from ding.model import DEC_MAVAC 
from net_config import main_config, create_config
from multi_env import NetworkEnv
import treetensor.numpy as tnp
import treetensor.torch as ttorch

def main(main_config: EasyDict, create_config: EasyDict):
    
    main_config.exp_name = 'decentralized_test'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    model_path = '../saved_models/<file_name>.pth.tar'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model weights not found at {}".format(model_path))
    env = NetworkEnv(cfg) 
    
    model = DEC_MAVAC(**cfg.policy.model)
    state_1 = torch.load(model_path, map_location='cpu') 
    state_2 = torch.load(model_path, map_location='cpu') 
    state_3 = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_1['model'],state_2['model'],state_3['model']) 
    policy = DecentralizedPolicy(cfg.policy, model=model).eval_mode 
    forward_fn = policy.forward 
    obs = env.reset() 
    
    n_episode = 2
    n_steps = 40 
    timestep_reward = []
    episode = 0
    while episode < n_episode: 
        
        returns = 0
        step = 0  
        while step < n_steps:
            if isinstance(obs, dict):
                obs = {i : tnp.array(o) for i,o in obs.items()}
            obs = tnp.stack(obs,axis=0)
            obs = ttorch.as_tensor(obs).to(dtype=ttorch.float32)
            obs = {0: obs} 
           
            data = forward_fn(obs)  
            action = data[0]['action']
            action = [action[i] for i in range(len(action))]
            print("Episode {}, Step {}, Action: {}".format(episode, step, action))
            obs, rew_n, done, info = env.step(action)
            rew = mean(rew_n)
            
            timestep_reward.append(rew)
            returns += rew 
            step+=1
            
        episode += 1
    

if __name__ == "__main__":

    main(main_config=main_config, create_config=create_config)
