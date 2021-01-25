from robogym.envs.push.blocks import BlockRearrangeEnv
from robogym.envs.push.common.base import RearrangeEnvConstants
from robogym.envs.push.goals.pickandplace import PickAndPlaceGoal
from robogym.envs.push.simulation.blocks import BlockRearrangeSim


class BlocksPickAndPlaceEnv(BlockRearrangeEnv):
    @classmethod
    def build_goal_generation(
        cls, constants: RearrangeEnvConstants, mujoco_simulation: BlockRearrangeSim
    ):
        return PickAndPlaceGoal(mujoco_simulation, constants.goal_args)


make_env = BlocksPickAndPlaceEnv.build
