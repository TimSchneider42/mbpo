from typing import Dict, List, Union, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .endeffector_velocity_controller import EndEffectorVelocityController
from .. import BaseTask
from ..ball_placing_task import BallPlacingTask


class BallPlacingEndEffectorVelocityController(EndEffectorVelocityController):
    def __init__(self, robot_name: str, linear_limits_lower: Union[np.ndarray, float],
                 linear_limits_upper: Union[np.ndarray, float], angular_limits_lower: Union[np.ndarray, float],
                 angular_limits_upper: Union[np.ndarray, float], smoothness_penalty_weight: float = 0.1,
                 rotation_range: float = 0.3, control_lin_accel: bool = False, max_lin_accel: float = 1.0):
        super().__init__(robot_name, linear_limits_lower, linear_limits_upper, angular_limits_lower,
                         angular_limits_upper, smoothness_penalty_weight=smoothness_penalty_weight)
        self.__robot_name = robot_name
        self.__rotation_range = rotation_range
        self.__control_lin_accel = control_lin_accel
        self.__max_lin_accel = max_lin_accel
        self._lin_vel_lim_lower = self._lin_vel_lim_upper = None

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        task = self.task
        dt = self.task.time_step
        assert isinstance(task, BallPlacingTask)
        gripper = self.robot.gripper
        current_pose = gripper.wrapped_body.links["tcp"].pose
        gripper_pose_table_frame = task.table_top_center_pose.transform(current_pose, inverse=True)
        gripper_vel_table_frame = task.table_top_center_pose.rotation.apply(gripper.velocity)
        gripper_lin_vel_table_frame, gripper_ang_vel_table_frame = gripper_vel_table_frame
        gripper_pos = gripper_pose_table_frame.translation[:2]
        lim = task.table_top_accessible_extents / 2 - task.ball_radius * 2 - 0.02

        if self.__control_lin_accel:
            current_vel = gripper_lin_vel_table_frame[:2]
            target_linear_vel_xy = current_vel + action[:2] * dt
            linear_vel_xy = np.clip(target_linear_vel_xy, self._lin_vel_lim_lower, self._lin_vel_lim_upper)
        else:
            linear_vel_xy = action[:2]
        # Ensure that gripper is not leaving operation area within the next time step
        linear_vel_xy_clipped = np.clip(linear_vel_xy, (-lim - gripper_pos) / dt, (lim - gripper_pos) / dt)
        angular_vel_z = action[2]
        current_z_rotation = gripper_pose_table_frame.rotation.as_euler("XYZ")[2]
        angular_vel_z_clipped = np.clip(angular_vel_z, (-self.__rotation_range - current_z_rotation) / dt,
                                        (self.__rotation_range - current_z_rotation) / dt)
        p_lin = 10.0
        d_lin = 0.0
        # PD controller to keep gripper at steady z coordinate
        linear_vel_z = -(gripper_pose_table_frame.translation[2] - task.gripper_distance_to_table) * p_lin \
                       - gripper_lin_vel_table_frame[2] * d_lin
        p_ang = 10.0
        d_ang = 0.0
        # This works only for small deltas
        # This code was brought to you by trial-and-error (TM)
        # Ensure that the gripper only rotates around the z axis
        target_orientation = Rotation.from_euler("xyz", np.array([np.pi, 0, 0]))
        current_rot_target_frame = target_orientation.inv() * gripper_pose_table_frame.rotation
        gripper_ang_vel_target_frame = target_orientation.apply(gripper_ang_vel_table_frame, inverse=True)
        difference_euler = current_rot_target_frame.as_euler("XYZ")
        angular_corr_vel_xy_target_frame = -difference_euler[:2] * p_ang - gripper_ang_vel_target_frame[:2] * d_ang
        angular_corr_vel_target_frame = np.concatenate([angular_corr_vel_xy_target_frame, [0]])
        angular_corr_vel_world_frame = target_orientation.apply(angular_corr_vel_target_frame)
        linear_vel_table_frame = np.concatenate([linear_vel_xy_clipped, [linear_vel_z]])
        # TODO: why is there a minus there?
        angular_vel_table_frame = angular_corr_vel_world_frame - np.array([0, 0, angular_vel_z_clipped])
        linear_vel_world_frame = task.table_top_center_pose.rotation.apply(linear_vel_table_frame)
        angular_vel_world_frame = task.table_top_center_pose.rotation.apply(angular_vel_table_frame)
        super(BallPlacingEndEffectorVelocityController, self)._actuate_denormalized(
            np.concatenate([linear_vel_world_frame, angular_vel_world_frame]))

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        limits_lower, limits_upper = super(BallPlacingEndEffectorVelocityController, self)._initialize(task)
        ang_lim_lower = limits_lower[5]
        ang_lim_upper = limits_upper[5]
        self._lin_vel_lim_lower = limits_lower[[0, 1]]
        self._lin_vel_lim_upper = limits_upper[[0, 1]]
        if self.__control_lin_accel:
            lin_lim_lower = np.full(2, -self.__max_lin_accel)
            lin_lim_upper = np.full(2, self.__max_lin_accel)
        else:
            lin_lim_lower = self._lin_vel_lim_lower
            lin_lim_upper = self._lin_vel_lim_upper
        return np.concatenate([lin_lim_lower, [ang_lim_lower]]), np.concatenate([lin_lim_upper, [ang_lim_upper]])

    @classmethod
    def from_parameters(
            cls, robot_name: str, parameters: Dict[str, List[float]], smoothness_penalty_weight: float = 0.1,
            rotation_range: float = 0.3, control_lin_accel: bool = False,
            max_lin_accel: float = 1.0) -> "BallPlacingEndEffectorVelocityController":
        """
        Create an BallPlacingEndEffectorVelocityController from an parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param parameters:                      a dictionary containing the entries end_effector_velocity_limits_lower
                                                and end_effector_velocity_limits_upper
        :param smoothness_penalty_weight:       TODO
        :return:                                an BallPlacingEndEffectorVelocityController with the given parameters
        """
        kwargs = {
            "{}_limits_{}".format(t, l): parameters["end_effector_{}_velocity_limits_{}".format(t, l)]
            for t in ["linear", "angular"]
            for l in ["upper", "lower"]
        }
        return BallPlacingEndEffectorVelocityController(
            robot_name, **kwargs, smoothness_penalty_weight=smoothness_penalty_weight, rotation_range=rotation_range,
            control_lin_accel=control_lin_accel, max_lin_accel=max_lin_accel)
