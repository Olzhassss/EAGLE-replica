from typing import Union, Dict, Optional
import torch
import torch.nn as nn
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY


@MODEL_REGISTRY.register('dec_mavac')
class DEC_MAVAC(nn.Module):
    r"""
    Overview:
        The DEC_MAVAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        agent_obs_shape: Union[int, SequenceType],
        global_obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        agent_num: int,
        actor_head_hidden_size: int = 256,
        actor_head_layer_num: int = 2,
        critic_head_hidden_size: int = 512,
        critic_head_layer_num: int = 1,
        action_space: str = 'discrete',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        bound_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the VAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - share_encoder (:obj:`bool`): Whether share encoder.
            - continuous (:obj:`bool`): Whether collect continuously.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for critic's nn.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(DEC_MAVAC, self).__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape, self.agent_obs_shape, self.action_shape = global_obs_shape, agent_obs_shape, action_shape
        self.action_space = action_space
        self.actor=nn.ModuleList()
        self.critic=nn.ModuleList()


        for i in range(agent_num):
            actor = nn.Sequential(
                nn.Linear(in_features=self.global_obs_shape, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=self.action_shape),
                nn.Softmax(dim=-1)
            )
            critic = nn.Sequential(
                nn.Linear(in_features=self.global_obs_shape, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=1)
            )
            for m in actor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.uniform_(m.weight, a=-0.04, b=0.04) 

            for m in critic.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

            self.actor.append(actor) 
            self.critic.append(critic)
        

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict output.
            Parameter updates with VAC's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'`` or ``'compute_critic'``:
                - inputs (:obj:`torch.Tensor`):
                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

                Forward with ``'compute_actor'``, Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.

                Forward with ``'compute_critic'``, Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N corresponding ``hidden_size``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Actor Examples:
            >>> model = VAC(64,128)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 128])

        Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> critic_outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)

        Actor-Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = actor_head_hidden_size``
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

        ReturnsKeys:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N, A)`, where B is batch size, N is num_agents, A is ``action_shape``

        """
        x = x['global_obs']  
        logit = []
        x = x.float()
        
        for i in range(len(self.actor)):
            y = self.actor[i](x)  
            logit.append(y)
        
        return {'logit': logit}

    def compute_critic(self, x: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`Dict`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = critic_head_hidden_size``
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

                Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - value (:obj:`torch.FloatTensor`): :math:`(B,N)`, where B is batch size, N num_agents.

        """
        x = x['global_obs'] 
        value = []
        x = x.float()

        for i in range(len(self.critic)):
            y = self.critic[i](x)
            value.append(y)
        
        return {'value': value}

    def compute_actor_critic(self, x: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The encoded embedding tensor.

        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.
            - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])


        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder.
            Returning the combination dictionry.

        """
        logit = self.compute_actor(x)['logit']
        value = self.compute_critic(x)['value']
        return {'logit': logit, 'value': value}

    def load_state_dict(self, x1: Dict, x2: Dict, x3: Dict) -> Dict:
        
        model1_actor_keys = [key for key in x1.keys() if key.startswith('actor.0')]
        model1_critic_keys = [key for key in x1.keys() if key.startswith('critic.0')]
        model2_actor_keys = [key for key in x2.keys() if key.startswith('actor.1')]
        model2_critic_keys = [key for key in x2.keys() if key.startswith('critic.1')]
        model3_actor_keys = [key for key in x3.keys() if key.startswith('actor.2')]
        model3_critic_keys = [key for key in x3.keys() if key.startswith('critic.2')]  

        actor_dict = {}
        critic_dict = {}
        model1_actor = {}
        model1_critic = {}
        model2_actor = {}
        model2_critic = {}
        model3_actor = {}
        model3_critic = {}

        for key in model1_actor_keys:
            model1_actor[key] = x1[key]
        for key in model1_critic_keys:
            model1_critic[key] = x1[key]
        
        for key in model2_actor_keys:
            model2_actor[key] = x2[key]
        for key in model2_critic_keys:
            model2_critic[key] = x2[key]
     
        for key in model3_actor_keys:
            model3_actor[key] = x3[key]
        for key in model3_critic_keys:
            model3_critic[key] = x3[key] 
        
        for d in [model1_actor, model2_actor, model3_actor]:
            for key, value in d.items():
                new_key = key.replace('actor.', '')
                actor_dict[new_key] = value
        for d in [model1_critic, model2_critic, model3_critic]:
            for key, value in d.items():
                new_key = key.replace('critic.', '')
                critic_dict[new_key] = value
        
        self.actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)

