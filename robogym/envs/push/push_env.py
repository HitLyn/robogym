import logging
import os
from typing import Dict, List
import numpy as np
import attr
from termcolor import cprint
from robogym.envs.push.common.mesh import (
    MeshRearrangeEnv,
    MeshRearrangeEnvConstants,
    MeshRearrangeEnvParameters,
)
from robogym.envs.push.common.utils import find_meshes_by_dirname
from robogym.envs.push.simulation.mesh import MeshRearrangeSim
from robogym.utils import rotation
from matplotlib import pyplot as plt
from datetime import datetime
from IPython import embed
logger = logging.getLogger(__name__)


def find_ycb_meshes() -> Dict[str, list]:
    return find_meshes_by_dirname("ycb")


def extract_object_name(mesh_files: List[str]) -> str:
    """
    Given a list of mesh file paths, this method returns an consistent name for the object

    :param mesh_files: List of paths to mesh files on disk for the object
    :return: Consistent name for the object based on the mesh file paths
    """
    dir_names = sorted(set([os.path.basename(os.path.dirname(p)) for p in mesh_files]))
    if len(dir_names) != 1:
        logger.warning(
            f"Multiple directory names found: {dir_names} for object: {mesh_files}."
        )

    return dir_names[0]


@attr.s(auto_attribs=True)
class YcbRearrangeEnvConstants(MeshRearrangeEnvConstants):
    # Whether to sample meshes with replacement
    sample_with_replacement: bool = True
    success_threshold: dict = {"obj_pos": 0.05, "obj_rot": 0.2}


class YcbRearrangeEnv(
    MeshRearrangeEnv[
        MeshRearrangeEnvParameters, YcbRearrangeEnvConstants, MeshRearrangeSim,
    ]
):
    MESH_FILES = find_ycb_meshes()

    def __init__(self, goal_type = 'pos', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cached_object_names: Dict[str, str] = {}
        push_candidates = ["077_rubiks_cube",
                           ]
        self.parameters.mesh_names = push_candidates
        self.goal_type = goal_type # from ['pos', 'goal', 'all']
        self.x_range = np.array([0.43, 0.85])
        self.y_range = np.array([0.40, 1.10])

    def _recreate_sim(self) -> None:
        # Call super to recompute `self.parameters.simulation_params.mesh_files`.
        super()._recreate_sim()

        # Recompute object names from new mesh files
        self._cached_object_names = {}
        for obj_group in self.mujoco_simulation.object_groups:
            mesh_obj_name = extract_object_name(obj_group.mesh_files)
            for i in obj_group.object_ids:
                self._cached_object_names[f"object{i}"] = mesh_obj_name

    def _sample_object_meshes(self, num_groups: int) -> List[List[str]]:
        if self.parameters.mesh_names is not None:
            candidates = [
                files
                for dir_name, files in self.MESH_FILES.items()
                if dir_name in self.parameters.mesh_names
            ]
        else:
            candidates = list(self.MESH_FILES.values())

        assert len(candidates) > 0, f"No mesh file for {self.parameters.mesh_names}."
        candidates = sorted(candidates)
        replace = self.constants.sample_with_replacement
        indices = self._random_state.choice(
            len(candidates), size=num_groups, replace=replace
        )

        return [candidates[i] for i in indices]

    def _get_simulation_info(self) -> dict:
        simulation_info = super()._get_simulation_info()
        simulation_info.update(self._cached_object_names)

        return simulation_info

    def clip_action(self, gripper_pos, action):
        # x direction
        if gripper_pos[0] + action[0]*0.07 >= self.x_range[1] or gripper_pos[0] + action[0]*0.07 <= self.x_range[0]:
            action[0] = 0
        if gripper_pos[1] + action[1]*0.07 >= self.y_range[1] or gripper_pos[1] + action[1]*0.07 <= self.y_range[0]:
            action[1] = 0
        return action

    def step(self, action):
        # check movtion range
        obs, reward, done, info = self.get_observation()
        gripper_pos = obs["gripper_pos"]
        clipped_action = self.clip_action(gripper_pos, action)
        full_action = np.zeros(5)
        full_action[:2] = clipped_action[:]
        obs, reward, done, info = super().step(full_action)
        # obs, reward, done, info = super().step(action)
        # embed()
        obs["observation"] = np.concatenate([obs["obj_pos"].squeeze().copy(), obs["obj_rot"].squeeze().copy(), obs["gripper_pos"].squeeze().copy()])
        obs["achieved_goal"] = np.concatenate([obs["obj_pos"].squeeze().copy(),])
        obs["desired_goal"] = np.concatenate([obs["goal_obj_pos"].squeeze().copy()])
        # obs["observation"] = np.concatenate([obs["obj_pos"].squeeze().copy(), obs["obj_rot"].squeeze().copy(), obs["gripper_pos"].squeeze().copy()])
        # obs["achieved_goal"] = np.concatenate([obs["obj_pos"].squeeze().copy(), obs["obj_rot"].squeeze().copy()])
        # obs["desired_goal"] = np.concatenate([obs["goal_obj_pos"].squeeze().copy(), obs["goal_obj_rot"].squeeze().copy()])
        obs["is_success"] = info["goal_achieved"]

        # cprint(obs["is_success"], "red")

        return obs, reward, done, info
    def reset(self):
        # cprint("env reset", "red")
        obs = super().reset()
        obs["observation"] = np.concatenate([obs["obj_pos"].squeeze().copy(), obs["obj_rot"].squeeze().copy(), obs["gripper_pos"].squeeze().copy()])
        obs["achieved_goal"] = np.concatenate([obs["obj_pos"].squeeze().copy()])
        obs["desired_goal"] = np.concatenate([obs["goal_obj_pos"].squeeze().copy(),])
        obs["is_success"] = False

        return obs

make_env = YcbRearrangeEnv.build

if __name__ == '__main__':
    # in /push/simulation/base.py: set_object_colors() to change target object color
    # from mujoco_py import GlfwContext
    import numpy as np
    # import matplotlib.pyplot as plt
    # GlfwContext(offscreen=True)  # Create a window to init GLFW.
    env = make_env()
    for n in range(300):
        env.reset()
        for j in range(6):
            env.step([-0.1, 0])
        for i in range(10):
            name = '/homeL/cong/Dataset/push_sim/' + str(n) + '_' + str(i)
            # now = datetime.now()
            # current_time = now.strftime("%H:%M:%S")
            # name = path + current_time
            array = env.render(mode="rgb_array")
            plt.imsave(name, array, format='png')
            # plt.show()
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            env.step([x, y, 0, 0, 0])
