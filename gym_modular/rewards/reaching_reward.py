import numpy as np

from .reward import Reward
from ..ball_placing_task import BallPlacingTask


class ReachingReward(Reward[BallPlacingTask]):
    def __init__(self, intermediate_time_step_scale: float = 1.0):
        super(ReachingReward, self).__init__("reaching_reward", intermediate_time_step_scale)

    def _calculate_reward_unnormalized(self) -> float:
        gripper_pose = self.task.environment.robots["ur10"].gripper.pose
        gripper_pose_table_frame = self.task.table_top_center_pose.transform(gripper_pose, inverse=True)
        ball_pos_target_frame = gripper_pose_table_frame.translation[:2] - self.task.target_zone_pos_table_frame
        ball_in_target_zone = np.all(np.logical_and(ball_pos_target_frame <= self.task.target_zone_extents / 2,
                                                    ball_pos_target_frame >= -self.task.target_zone_extents / 2))
        return 1.0 if ball_in_target_zone else 0.0

    def _get_min_reward_unnormalized(self) -> float:
        return 0

    def _get_max_reward_unnormalized(self) -> float:
        return 1
