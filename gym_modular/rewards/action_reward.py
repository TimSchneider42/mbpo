import numpy as np

from .reward import Reward
from ..ball_placing_task import BallPlacingTask


class ActionReward(Reward[BallPlacingTask]):
    def __init__(self, intermediate_time_step_scale: float = 0.001):
        super(ActionReward, self).__init__("ball_placing_reward", intermediate_time_step_scale)

    def _calculate_reward_unnormalized(self) -> float:
        action = self.task.latest_action
        action_arr = np.concatenate(list(action.values()))
        return -(action_arr ** 2).sum()

    def _get_min_reward_unnormalized(self) -> float:
        action_names = list(self.task.action_space.spaces.keys())
        action_lims_upper = np.concatenate([self.task.action_space[n].high for n in action_names])
        action_lims_lower = np.concatenate([self.task.action_space[n].low for n in action_names])
        min_squared_value = np.minimum(action_lims_upper ** 2, action_lims_lower ** 2)
        zero_in_interval = np.logical_and(action_lims_lower <= 0, action_lims_upper >= 0)
        min_squared_value[zero_in_interval] = 0
        return -min_squared_value.sum()

    def _get_max_reward_unnormalized(self) -> float:
        action_names = list(self.task.action_space.spaces.keys())
        action_lims_upper = np.concatenate([self.task.action_space[n].high for n in action_names])
        action_lims_lower = np.concatenate([self.task.action_space[n].low for n in action_names])
        max_squared_value = np.maximum(action_lims_upper**2, action_lims_lower**2)
        return max_squared_value.sum()
