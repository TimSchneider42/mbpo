from abc import ABC
from typing import Dict, Tuple

import numpy as np

from .continuous_sensor import ContinuousSensor
from ..ball_placing_task import BallPlacingTask


class BallPositionSensor(ContinuousSensor[BallPlacingTask], ABC):
    def __init__(self):
        super(BallPositionSensor, self).__init__(clip=True)

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        te = self.task.table_top_accessible_extents
        return {"ball_position": (-te / 2, te / 2)}

    def __observe(self) -> Dict[str, np.ndarray]:
        return {
            "ball_position": self.task.table_top_center_pose.transform(
                self.task.ball.pose, inverse=True).translation[:2]}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()
