from robogym.envs.rearrange_t.blocks import make_env

env = make_env(
    parameters={
        'simulation_params': {
            'num_objects': 5,
            'max_num_objects': 8,
        }
    }
)
obs = env.reset()
while True:
    env.render()