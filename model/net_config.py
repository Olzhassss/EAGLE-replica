from easydict import EasyDict

n_agent = 3
collector_env_num = 1
evaluator_env_num = 1

net_collaq_cfg = dict(
    exp_name='decentralized_tr',
    env=dict(
        env_family='network',
        n_agent=n_agent,
        max_steps=20, 
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
            max_retry=1,
        ),
    ),
    policy=dict(
        cuda=False,
        model=dict(
            agent_num=n_agent,
            agent_obs_shape=4,
            global_obs_shape=12,
            action_shape=2,
        ),
        agent_num=n_agent,
        learn=dict(
            multi_gpu=False,
            update_per_collect=1,
            batch_size=20, 
            learning_rate_critic=0.001,
            learning_rate_actor=0.0001,
            clip_value=5,
        ),
        collect=dict(
            unroll_len=1,
            n_sample=20, 
        ),
        eval=dict(env_num=evaluator_env_num,evaluator=dict(eval_freq=30, )),  
        other=dict(
            eps=dict(
            type='exp',
            start=1.0,
            end=0.05,
            decay=100000,
        ), 
            replay_buffer=dict(
            replay_buffer_size=20, 
            ),
        ),
    ),
)
net_collaq_cfg = EasyDict(net_collaq_cfg)
main_config = net_collaq_cfg
net_create_config = dict(
    env=dict(
        type='multi_network',
        import_names=['multi_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='decentralized'),
    collector=dict(type='episode', get_train_sample=True),
)
net_create_config = EasyDict(net_create_config)
create_config = net_create_config

def train(args):
    config = [main_config, create_config]
    serial_pipeline(config, seed=args.seed)

if __name__ == "__main__":
    
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
