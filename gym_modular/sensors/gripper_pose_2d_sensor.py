from typing import Dict, Tuple

import numpy as np

from .continuous_sensor import ContinuousSensor
from ..ball_placing_task import BallPlacingTask


class GripperPose2DSensor(ContinuousSensor[BallPlacingTask]):
    def __init__(self, robot_name: str = "ur10", rotation_range: float = 0.3):
        assert 0 <= rotation_range <= np.pi / 4
        super(GripperPose2DSensor, self).__init__()
        self.__robot_name = robot_name
        self.__rotation_range = rotation_range

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        lim: np.ndarray = self.task.table_top_accessible_extents / 2 - self.task.ball_radius * 2 - 0.02
        if self.__rotation_range > 0.0:
            return {
                "gripper_pos_xy": (-lim, lim),
                "gripper_angle_z": (-np.array([self.__rotation_range]), np.array([self.__rotation_range]))
            }
        else:
            return {
                "gripper_pos_xy": (-lim, lim)
            }

    def __observe(self) -> Dict[str, np.ndarray]:
        gripper_pose = self.task.environment.robots["ur10"].gripper.wrapped_body.links["tcp"].pose
        gripper_pose_table_frame = self.task.table_top_center_pose.transform(gripper_pose, inverse=True)
        angle = gripper_pose_table_frame.rotation.as_euler("XYZ")[2]
        pos_xy = gripper_pose_table_frame.translation[:2]
        if self.__rotation_range > 0.0:
            return {
                "gripper_pos_xy": pos_xy,
                "gripper_angle_z": angle
            }
        else:
            return {
                "gripper_pos_xy": pos_xy
            }

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()
