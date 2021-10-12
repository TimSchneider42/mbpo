from typing import Tuple

import numpy as np

from assembly_gym.environment.generic import JointMode
from assembly_gym.environment.pybullet import PyBulletRobot
from assembly_gym.util import Transformation
from .. import BaseTask
from .robot_component_controller import RobotComponentController
from ..ball_placing_task import BallPlacingTask


# Warning: this controller seems to have problems with rotations of the gripper in place
class BallPlacingEndEffectorPositionController(RobotComponentController):
    def __init__(self, robot_name: str, max_pos_step: float = 0.01, max_angle_step: float = 0.03,
                 rotation_range: float = 0.3):
        assert 0 <= rotation_range <= np.pi / 4
        super().__init__("bp_ee_pos_controller", robot_name, "arm")
        self.__max_pos_step = max_pos_step
        self.__max_angle_step = max_angle_step
        self.__rotation_range = rotation_range

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        task = self.task
        assert isinstance(task, BallPlacingTask)
        table_pose = task.table_top_center_pose
        pos_delta = action[:2]
        robot = self.robot
        assert isinstance(robot, PyBulletRobot)
        current_pose = robot.gripper.wrapped_body.links["tcp"].pose
        current_pose_table_frame = table_pose.transform(current_pose, inverse=True)
        target_position_xy = current_pose_table_frame.translation[:2] + pos_delta
        lim = task.table_top_accessible_extents / 2 - task.ball_radius * 2 - 0.02
        target_position_xy_clipped = np.clip(target_position_xy, -lim, lim)
        target_position_table_frame = np.array(
            [target_position_xy_clipped[0], target_position_xy_clipped[1], task.gripper_distance_to_table])
        if self.__rotation_range > 0.0:
            angle_delta = action[2]
            target_angle_unclipped = current_pose_table_frame.euler[2] + angle_delta
            target_angle = np.clip(target_angle_unclipped, -self.__rotation_range, self.__rotation_range)
        else:
            target_angle = 0.0
        target_pose_table_frame = Transformation.from_pos_euler(
            position=target_position_table_frame,
            euler_angles=[np.pi, 0, target_angle])
        target_pose_world_frame = table_pose.transform(target_pose_table_frame)
        joint_angles = robot.solve_ik(target_pose_world_frame)
        robot.arm.set_joint_target_positions(joint_angles)

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        self.robot.arm.set_joint_mode(JointMode.POSITION_CONTROL)
        low = np.array([-self.__max_pos_step, -self.__max_pos_step, -self.__max_angle_step])
        high = np.array([self.__max_pos_step, self.__max_pos_step, self.__max_angle_step])
        if self.__rotation_range > 0.0:
            return low, high
        else:
            return low[:2], high[:2]
