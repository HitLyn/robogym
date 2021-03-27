from push import make_env
from mujoco_py import GlfwContext
import numpy as np
import matplotlib.pyplot as plt


GlfwContext(offscreen=True)
env = make_env()
env.reset()

for i in range(5):
    name = '/home/lyn/' + str(i)
    with env.unwrapped.mujoco_simulation.hide_objects():
        array = env.render(mode="rgb_array")
        plt.imsave(name, array, format = 'png')
        env.step([0.5, 0.5, 0,0,0])

for i in range(5):
    name = '/home/lyn/' + str(i) + '_'
    array = env.render(mode="rgb_array")
    plt.imsave(name, array, format = 'png')
    env.step([-0.5,-0.5, 0,0,0])
