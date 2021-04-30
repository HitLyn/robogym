import logging
import os
from typing import Dict, List

import attr

from robogym.envs.push.common.mesh import (
    MeshRearrangeEnv,
    MeshRearrangeEnvConstants,
    MeshRearrangeEnvParameters,
)
from robogym.envs.push.common.utils import find_meshes_by_dirname
from robogym.envs.push.simulation.mesh import MeshRearrangeSim
from matplotlib import pyplot as plt
from datetime import datetime
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
    success_threshold: dict = {"obj_pos": 0.02, "obj_rot": 0.15}


class YcbRearrangeEnv(
    MeshRearrangeEnv[
        MeshRearrangeEnvParameters, YcbRearrangeEnvConstants, MeshRearrangeSim,
    ]
):
    MESH_FILES = find_ycb_meshes()

    def __init__(self, goal_type = 'pos', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cached_object_names: Dict[str, str] = {}
        # push_candidates = ["004_sugar_box", "011_banana", "013_apple", "024_bowl", "030_fork", "032_knife", "037_scissors", "031_spoon", "035_power_drill",
        #                    "044_flat_screwdriver", "048_hammer", "072-a_toy_airplane", "077_rubiks_cube", "065-a_cups", "073-a_lego_duplo", "065-f_cups", "033_spatula",
        #                    "029_plate", "025_mug", "027_skillet"]
        push_candidates = ["len_8_block", "r_40_cylinder", "len_10_block", "len_6_block", "r_30_cylinder"]
        self.parameters.mesh_names = push_candidates

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


make_env = YcbRearrangeEnv.build

if __name__ == '__main__':
    # in /push/simulation/base.py: set_object_colors() to change target object color
    from mujoco_py import GlfwContext
    import numpy as np
    import matplotlib.pyplot as plt
    GlfwContext(offscreen=True)  # Create a window to init GLFW.
    env = make_env()
    for n in range(1000):
        env.reset()
        for j in range(6):
            env.step([-0.5, 0, 0, 0, 0])
        for i in range(10):
            name = '/homeL/cong/HitLyn/Visual-Pushing/images/red_original/' + "{:0>5d}.png".format(10*n + i)
            # now = datetime.now()
            # current_time = now.strftime("%H:%M:%S")
            # name = path + current_time
            array = env.render(mode="rgb_array")
            plt.imsave(name, array, format='png')
            # plt.show()
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            env.step([x, y, 0, 0, 0])
