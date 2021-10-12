from typing import Dict, Tuple, List

import numpy as np

from .continuous_sensor import ContinuousSensor
from ..ball_placing_task import BallPlacingTask


class GripperVelocity2DSensor(ContinuousSensor[BallPlacingTask]):
    def __init__(self, robot_name: str, linear_limit_lower: np.ndarray, linear_limit_upper: np.ndarray,
                 angular_limit_lower: float, angular_limit_upper: float, sense_angle: bool = True):
        super(GripperVelocity2DSensor, self).__init__()
        self.__robot_name = robot_name
        self.__sense_angle = sense_angle
        self.__linear_limit_lower = linear_limit_lower
        self.__linear_limit_upper = linear_limit_upper
        self.__angular_limit_lower = angular_limit_lower
        self.__angular_limit_upper = angular_limit_upper

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        lin_lims = (self.__linear_limit_lower, self.__linear_limit_upper)
        ang_lims = (np.array([self.__angular_limit_lower]), np.array([self.__angular_limit_upper]))
        if self.__sense_angle:
            return {
                "gripper_vel_xy": lin_lims,
                "gripper_angular_vel_z": ang_lims
            }
        else:
            return {
                "gripper_vel_xy": lin_lims
            }

    def __observe(self) -> Dict[str, np.ndarray]:
        vel = self.task.environment.robots["ur10"].gripper.wrapped_body.links["tcp"].velocity
        lin_table_frame, ang_table_frame = self.task.table_top_center_pose.rotation.apply(vel, inverse=True)
        if self.__sense_angle:
            return {
                "gripper_vel_xy": lin_table_frame[:2],
                "gripper_angular_vel_z": ang_table_frame[2:3]
            }
        else:
            return {"gripper_vel_xy": lin_table_frame[:2]}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    @classmethod
    def from_parameters(cls, robot_name: str, parameters: Dict[str, List[float]],
                        sense_angle: bool = True) -> "GripperVelocity2DSensor":
        """
        Create an BallPlacingEndEffectorVelocityController from an parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param parameters:                      a dictionary containing the entries end_effector_velocity_limits_lower
                                                and end_effector_velocity_limits_upper
        :return:                                an BallPlacingEndEffectorVelocityController with the given parameters
        """
        linear_lower = parameters["end_effector_linear_velocity_limits_lower"]
        linear_upper = parameters["end_effector_linear_velocity_limits_upper"]
        angular_lower = parameters["end_effector_angular_velocity_limits_lower"]
        angular_upper = parameters["end_effector_angular_velocity_limits_upper"]
        assert np.all(np.equal(angular_lower, angular_lower[0])) and \
               np.all(np.equal(angular_upper, angular_upper[0])) and \
               angular_lower[0] == -angular_upper[0], "Differing angular limits are not supported"
        assert np.all(np.equal(linear_lower, linear_lower[0])) and \
               np.all(np.equal(linear_upper, linear_upper[0])) and \
               linear_lower[0] == -linear_upper[0], "Differing linear limits are not supported"
        lin = linear_upper[0]
        ang = angular_upper[0]
        return GripperVelocity2DSensor(
            robot_name, linear_limit_lower=np.full(2, -lin), linear_limit_upper=np.full(2, lin),
            angular_limit_lower=-ang, angular_limit_upper=ang, sense_angle=sense_angle)
