from typing import List, Dict, Any, Tuple
from collections import namedtuple
import torch,time
import copy
import torch.nn as nn
from ding.torch_utils import Adam, to_device
from ding.rl_utils import  get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
import torch.nn.utils.prune as pruning


@POLICY_REGISTRY.register('decentralized')
class DecentralizedPolicy(Policy):
    r"""
    Overview:
        Policy class of on policy version PPO algorithm.
    """
    config = dict(
        type='fully_dec',
        cuda=False,
        on_policy=False, 
        priority=False,
        priority_IS_weight=False,
        action_space='discrete',
        nstep_return=False,
        multi_agent=True,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            update_per_collect=1,
            batch_size=64,
            learning_rate=0.0001,
            ignore_done=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=64,
            unroll_len=1,
        ),
        eval=dict(),
        other=dict(
            eps=dict(
                # (str) Type of epsilon decay
                type='exp',
                # (float) Start value for epsilon decay, in [0, 1].
                # 0 means not use epsilon decay.
                start=1,
                # (float) Start value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Decay length(env step)
                decay=200000,
            ),
            replay_buffer=dict(
                # (int) max size of replay buffer
                replay_buffer_size=5000,
                max_reuse=10,
            ),
        ), 
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main model.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in fully_dec"

        assert self._cfg.action_space in ["discrete"]
        self._action_space = self._cfg.action_space
        self._agent_num = self._cfg.agent_num
        self._input_space = self._cfg.model.global_obs_shape
        self._estimator_shape =  self._cfg.model.agent_obs_shape
        self._optimizers_critic = []
        self._optimizers_actor = []
        self._reward_approx = []
        self._reward_opt = []
        self._mu = []
        self._iter = 1
        self.to_prune = False
        self.amount = 0.3

        for i in range(self._agent_num):
        
            optimizer_actor = Adam(
                self._model.actor[i].parameters(),
                lr=self._cfg.learn.learning_rate_actor,
            )
            optimizer_critic = Adam(
                self._model.critic[i].parameters(),
                lr=self._cfg.learn.learning_rate_critic,
            )

            R = nn.Sequential(
                nn.Linear(in_features=self._input_space + 3, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(), 
                nn.Linear(in_features=64, out_features=1)
            )

            for m in R.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            
            R_opt = Adam(
                R.parameters(),
                lr=0.001, 
            )

            mu = torch.nn.Parameter(torch.zeros(1))

            self._optimizers_actor.append(optimizer_actor)
            self._optimizers_critic.append(optimizer_critic)
            self._reward_approx.append(R)
            self._reward_opt.append(R_opt)
            self._mu.append(mu)

        self._learn_model = model_wrap(self._model, wrapper_name='base') 

        self._beta_critic = self._cfg.learn.learning_rate_critic
        self._beta_actor = self._cfg.learn.learning_rate_actor
        self._beta_mu = 0
        self._mean_mu = 0

        # Main model
        self._learn_model.reset()

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs','value']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        self._v_tilde = []        
        self._lambda_tilde = []
        self._iter = self._iter+1 


        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        self._beta_mu = 1 / pow(self._iter,0.85)

        # ====================
        # critic learn forward
        # ====================
        
        value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
        with torch.no_grad():
                next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value'] 
                
        # ====================
        # actor learn forward
        # ====================
        actor_logit = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']

        reward = torch.transpose(data['reward'],0,1)
        action = torch.transpose(data['action'],0,1)    

        loss_critic,loss_avg_reward,loss_actor = [],[],[]
        rew_predicts=[]
        
        for i in range(self._agent_num):
            mu_local = (1 - self._beta_mu)*self._mu[i] + self._beta_mu * reward[i]

            reward_reshaped = reward[i].float().unsqueeze(1).expand_as(value[i])
            with torch.no_grad():
                target = reward_reshaped - self._mu[i] + next_value[i] 
            critic = torch.nn.functional.mse_loss(value[i],target)

            action_all = data['action']*100            
            state_input = data['next_obs']['global_obs'].float()
            
            inputs = torch.cat((state_input,action_all), dim=1) 


            avg_reward = self._reward_approx[i](inputs)
            reward_i = reward[i].unsqueeze(1)
            delta_lambda = torch.nn.functional.mse_loss(avg_reward,reward_i.float())
            
            
            # ====================
            # critic update
            # ====================
             
            self._optimizers_critic[i].zero_grad()
            critic.backward() 
            self._optimizers_critic[i].step()
            # ====================
            # reward estimator update
            # ====================

            self._reward_opt[i].zero_grad()
            delta_lambda.backward()
            self._reward_opt[i].step()

            loss_critic.append(critic.data)
            loss_avg_reward.append(delta_lambda.data)
            rew_predicts.append(avg_reward)

        # ==============================================
        # actor update after all did critic update
        # ==============================================
        for i in range(self._agent_num): 
            
            est_all=torch.transpose(torch.stack(rew_predicts),0,1)
            mean_est = torch.mean(est_all,dim=1)

            v_prime = next_value[i].expand_as(mean_est)
            v = value[i].expand_as(mean_est)

            dist = torch.distributions.categorical.Categorical(probs=actor_logit[i])
            logp = dist.log_prob(action[i])   
            delta_policy =  mean_est - self._mean_mu  + v_prime - v
            delta_policy = delta_policy.squeeze()
            policy = -torch.mean(logp * delta_policy.detach())

            self._optimizers_actor[i].zero_grad()
            policy.backward()
            self._optimizers_actor[i].step()

            self._mu[i].data.copy_(mu_local.mean()) 

        # =============
        # consensus update
        # =============

        critic_params = [list(self._learn_model.critic[i].parameters()) for i in range(self._agent_num)]
        mean_critic_params = [torch.mean(torch.stack(param_list), dim=0) for param_list in zip(*critic_params)] 

        mus = []
        for j in range(self._agent_num):
            mus.append(self._mu[j].detach())
        self._mean_mu = torch.mean(torch.stack(mus))
       
        if self._iter > 5 and self.to_prune == True:
            # Apply L1 unstructured pruning to the weights of the critic
            tot_w = []   
            for i in range(self._agent_num) :
                tmp_r = copy.deepcopy(self._learn_model.critic[i])
                for layer,idx in enumerate(tmp_r):
                    if isinstance(layer, torch.nn.Linear) and idx > 0:  # Ensure the layer supports pruning
                        pruning.l1_unstructured(layer, name='weight', amount=self.amount)
                tot_w.append(tmp_r)
            critic_params = [list(tot_w[i].parameters()) for i in range(self._agent_num)]
            mean_critic_params = [torch.mean(torch.stack(param_list), dim=0) for param_list in zip(*critic_params)]  
            self.amount = self.amount + 0.05
            if self.amount > 0.7 : self.amount = 0.7
        
        for i in range(self._agent_num) :
            
            names = [name for name,param in self._learn_model.critic[i].named_parameters()]
            params = list(self._learn_model.critic[i].parameters())
            new_params = {}

            for j, param in enumerate(params):
                
                param.data = mean_critic_params[j].data.clone()
                new_params[names[j]] = param 

            self._learn_model.critic[i].load_state_dict(new_params) 

        return {
            'mu': self._mu[0],
        }
        
    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        critic_opt = []
        actor_opt = []  
        reward_approx = [] 
        reward_opt = []

        for i in range(self._agent_num):
            critic_opt.append(self._optimizers_critic[i].state_dict())
            actor_opt.append(self._optimizers_actor[i].state_dict())
            reward_approx.append(self._reward_approx[i].state_dict())
            reward_opt.append(self._reward_opt[i].state_dict())

        return {
            'model': self._learn_model.state_dict(),
            'optimizer_critic': critic_opt,
            'optimizer_actor': actor_opt,
            'reward_approximator': reward_approx,
            'reward_optimizer': reward_opt,
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.
        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        actor_opt = state_dict['optimizer_actor'] 
        critic_opt = state_dict['optimizer_critic']

        for i in range(self._agent_num):
            self._optimizers_actor[i].load_state_dict(actor_opt[i])
            self._optimizers_critic[i].load_state_dict(critic_opt[i])
            

    def _load_state_dict_reward(self, state_dict: List[Dict[str, Any]]) -> None:
        
        reward_approx = state_dict['reward_approximator']
        reward_opt = state_dict['reward_optimizer']

        for i in range(self._agent_num):
            self._reward_approx[i].load_state_dict(reward_approx[i])
            self._reward_opt[i].load_state_dict(reward_opt[i])
        
    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """

        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample') #no eps 
        self._collect_model.reset()
        
    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """ 
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data,mode='compute_actor_critic')
            action = torch.cat(output['action'])
            action = action.numpy()
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)} 
        
    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly.
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): List of training samples.
        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procedure by overriding this two methods and collector \
            itself.
        """
        return get_train_sample(data, self._unroll_len)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'mavac', ['ding.model.template.mavac']
     
     
    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()
        
    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor_critic')
            action = torch.cat(output['action'])
            action=action.numpy()
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}


    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max', 'grad_norm']    
