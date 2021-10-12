from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile

from assembly_gym.environment.simulation import SimulationObject, ShapeTypes, SimulationRobot
from typing import Tuple, Dict, Iterable, Optional, Sequence, TYPE_CHECKING, List
import numpy as np

from util.rectangle import Rectangle
from .controllers.controller import Controller
from .sensors import Sensor
from assembly_gym.util import Transformation
from .simulated_task import SimulatedTask

if TYPE_CHECKING:
    from .rewards.reward import Reward


class BallPlacingTask(SimulatedTask):
    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["BallPlacingTask"]], rewards: Iterable["Reward[BallPlacingTask]"],
                 time_step: float = 0.005, time_limit_steps: Optional[int] = None, use_ball: bool = True,
                 gripper_width: float = 0.02, ball_friction: float = 1.0, ball_mass: float = 0.02,
                 holes: Sequence[Rectangle] = ()):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param rewards:                 A sequence of rewards objects that provide rewards to the agent
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        super(BallPlacingTask, self).__init__(
            controllers, sensors, rewards, time_step, time_limit_steps=time_limit_steps)
        self.__ball: Optional[SimulationObject] = None
        self.__target_zone_extents = np.array([0.08, 0.05])
        self.__target_entry_depth = 0.02
        self.__table_extents = np.array([0.5, 0.5 + self.__target_zone_extents[1] + self.__target_entry_depth, 0.71])
        self.__table_pose = Transformation.from_pos_euler(position=np.array([0, 0.85, self.__table_extents[-1] / 2]),
                                                          euler_angles=np.array([0.2, 0.0, 0.0]))
        self.__table_top_center = self.__table_pose * Transformation.from_pos_euler(
            position=[0, 0, self.__table_extents[-1] / 2])
        self.__target_zone_height = 0.001
        self.__barrier_width = 0.02
        self.__barrier_height = 0.03
        self.__target_zone_pose_table_frame = Transformation.from_pos_euler(
            position=[0, (self.__table_extents[1] - self.__target_zone_extents[1] - 0.05) / 2 - self.__barrier_width,
                      self.__target_zone_height / 2])
        self.__target_zone_pos_table_frame = self.__target_zone_pose_table_frame.translation[:2]
        self.__ball_radius = 0.02
        self.__robot: Optional[SimulationRobot] = None
        self.__table_height = 0.71
        self.__gripper_z_dist_to_ball_center = 0.005
        self.__gripper_distance_to_table = self.__ball_radius + self.__gripper_z_dist_to_ball_center
        self.__use_ball = use_ball
        self.__gripper_width = gripper_width
        self.__ball_friction = ball_friction
        self.__holes = holes
        self.__hole_depth = self.__ball_radius
        self.__ball_mass = ball_mass
        self.initial_ball_pos_override: Optional[np.ndarray] = None

    def _add_box(self, extents: Sequence[float], pose: Transformation,
                 rgba_colors: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        collision_shape = self.environment.create_collision_shape(
            ShapeTypes.BOX, box_extents=[extents])
        visual_shape = self.environment.create_visual_shape(
            ShapeTypes.BOX, box_extents=[extents], rgba_colors=rgba_colors)
        box = self.environment.add_simple_object(visual_shape, collision_shape)
        box.set_pose(pose)

    def _initialize_scene(self) -> None:
        with (Path(__file__).parent / "cuboid_gripper_template.urdf").open() as f:
            gripper_urdf_template = f.read()
        gripper_urdf_data = gripper_urdf_template.format(gripper_width=self.__gripper_width)
        with NamedTemporaryFile("w", suffix=".urdf") as f:
            f.write(gripper_urdf_data)
            f.flush()
            self.__robot = self.environment.add_ur10_robot("ur10", rh_p12_rn_urdf_path=Path(f.name))
        self.__robot.set_pose(Transformation.from_pos_euler(position=np.array([0.0, 0.0, 0.71])))
        # Robot socket
        self._add_box([0.15, 0.15, 0.71], Transformation.from_pos_euler(position=np.array([0, 0, 0.355])),
                      rgba_colors=(0.7, 0.7, 0.7, 1.0))

        # Table
        bw = self.__barrier_width
        bh = self.__barrier_height
        table_base_color = (0.4, 0.3, 0.3, 1.0)
        surface_box_color = (0.5, 0.4, 0.4, 1.0)
        boundary_color = (0.3, 0.2, 0.2, 1.0)

        table_extents = self.__table_extents
        table_box_extents = table_extents.copy()
        if len(self.__holes) > 0:
            table_box_extents[2] -= self.__hole_depth
            table_center_top_frame = Transformation(
                translation=np.array([0.0, 0.0, -table_box_extents[2] / 2 - self.__hole_depth]))
            table_pose = self.__table_top_center.transform(table_center_top_frame)
            inverted_holes = self._invert_holes(self.__holes)
            for rect in inverted_holes:
                extents = np.concatenate([rect.max_coords - rect.min_coords, [self.__hole_depth]])
                pose_tt_center_frame = Transformation(
                    np.concatenate([(rect.max_coords + rect.min_coords) / 2, [-self.__hole_depth / 2]]))
                pose_world_frame = self.__table_top_center.transform(pose_tt_center_frame)
                self._add_box(extents, pose_world_frame, rgba_colors=surface_box_color)
            self._add_box(
                [table_extents[0], bw, bh],
                self.__table_top_center * Transformation.from_pos_euler(
                    position=[0, -(table_extents[1] / 2 - bw / 2), bh / 2 - self.__hole_depth]),
                rgba_colors=table_base_color)
            self._add_box(
                [table_extents[0], bw, bh],
                self.__table_top_center * Transformation.from_pos_euler(
                    position=[0, table_extents[1] / 2 - bw / 2, bh / 2 - self.__hole_depth]),
                rgba_colors=table_base_color)
            self._add_box(
                [bw, table_extents[1] - 2 * bw, bh],
                self.__table_top_center * Transformation.from_pos_euler(
                    position=[-(table_extents[0] / 2 - bw / 2), 0, bh / 2 - self.__hole_depth]),
                rgba_colors=table_base_color)
            self._add_box(
                [bw, table_extents[1] - 2 * bw, bh],
                self.__table_top_center * Transformation.from_pos_euler(
                    position=[table_extents[0] / 2 - bw / 2, 0, bh / 2 - self.__hole_depth]),
                rgba_colors=table_base_color)
        else:
            table_pose = self.__table_pose
        self._add_box(table_box_extents, table_pose, rgba_colors=table_base_color)

        # Boundaries
        self._add_box(
            [table_extents[0], bw, bh],
            self.__table_top_center * Transformation.from_pos_euler(
                position=[0, -(table_extents[1] / 2 - bw / 2), bh / 2]),
            rgba_colors=boundary_color)
        self._add_box(
            [table_extents[0], bw, bh],
            self.__table_top_center * Transformation.from_pos_euler(
                position=[0, table_extents[1] / 2 - bw / 2, bh / 2]),
            rgba_colors=boundary_color)
        self._add_box(
            [bw, table_extents[1] - 2 * bw, bh],
            self.__table_top_center * Transformation.from_pos_euler(
                position=[-(table_extents[0] / 2 - bw / 2), 0, bh / 2]),
            rgba_colors=boundary_color)
        self._add_box(
            [bw, table_extents[1] - 2 * bw, bh],
            self.__table_top_center * Transformation.from_pos_euler(
                position=[table_extents[0] / 2 - bw / 2, 0, bh / 2]),
            rgba_colors=boundary_color)

        # Barrier
        # self._add_box(
        #     [table_extents[0] - 2 * bw - 2 * tzw, bw, bh],
        #     self.__table_top_center * Transformation.from_pos_euler(
        #         position=[0, table_extents[1] / 2 - hb_extents[1] - tzw - bw / 2, bh / 2]),
        #     rgba_colors=b_color)

        if self.__use_ball:
            ball_collision_shape = self.environment.create_collision_shape(
                ShapeTypes.SPHERE, sphere_radii=[self.__ball_radius])
            ball_visual_shape = self.environment.create_visual_shape(
                ShapeTypes.SPHERE, sphere_radii=[self.__ball_radius], rgba_colors=(0.7, 0.2, 0.2, 1.0))
            self.__ball = self.environment.add_simple_object(
                ball_visual_shape, ball_collision_shape, mass=self.__ball_mass, friction=self.__ball_friction)
            self.__ball.set_pose(
                self.__table_top_center * Transformation.from_pos_euler(position=[0, 0, self.__ball_radius + 0.001]))

        target_zone_marker_visual = self.environment.create_visual_shape(
            ShapeTypes.BOX, box_extents=[list(self.__target_zone_extents) + [self.__target_zone_height]],
            rgba_colors=(1.0, 0.0, 0.0, 0.2))
        target_zone_marker = self.environment.add_simple_object(target_zone_marker_visual)
        target_zone_marker.set_pose(self.__table_top_center * self.__target_zone_pose_table_frame)

        gripper_pose_table_top_frame = Transformation.from_pos_euler(
            position=np.array([0, 0, self.__gripper_distance_to_table]), euler_angles=[np.pi, 0, 0])
        gripper_pose_tcp_world_frame = self.__table_top_center.transform(gripper_pose_table_top_frame)
        self.__robot.place_gripper_at(gripper_pose_tcp_world_frame)

    def _step_task(self) -> Tuple[bool, Dict]:
        return False, {"dummy": np.zeros(2)}

    def _reset_task(self) -> None:
        if self.initial_ball_pos_override is None:
            initial_ball_pos = np.random.uniform([-0.1, -0.15], [0.1, -0.15])
        else:
            initial_ball_pos = self.initial_ball_pos_override
            self.initial_ball_pos_override = None
        initial_ball_pos_table_frame = np.array([0.0, 0.0, self.__ball_radius + 0.001])
        initial_ball_pos_table_frame[:2] += initial_ball_pos
        initial_ball_pose_table_frame = Transformation.from_pos_euler(position=initial_ball_pos_table_frame)
        initial_ball_pose_world_frame = self.__table_top_center * initial_ball_pose_table_frame
        if self.__use_ball:
            self.__ball.set_pose(initial_ball_pose_world_frame)
        gripper_pose_ball_frame = Transformation.from_pos_euler(
            position=np.array([0, -0.03, self.__gripper_z_dist_to_ball_center]), euler_angles=np.array([np.pi, 0, 0]))
        gripper_pose_world_frame = initial_ball_pose_world_frame.transform(gripper_pose_ball_frame)
        self.__robot.place_gripper_at(gripper_pose_world_frame)

    def _invert_holes(self, holes: Sequence[Rectangle]) -> "List[Rectangle]":
        assert all(
            np.all(h.min_coords >= -1) and np.all(h.max_coords <= 1) and np.all(h.min_coords < h.max_coords)
            for h in holes), "Hole coordinates must be in [-1, 1]^2"
        assert all(
            not h1.intersects(h2, strict=True) for i, h1 in enumerate(holes) for h2 in
            holes[i + 1:]), "Holes must not intersect"
        cells_x = np.array(sorted({p for w in holes for p in [w.min_coords[0], w.max_coords[0]]}.union([-1, 1])))
        cells_y = np.array(sorted({p for w in holes for p in [w.min_coords[1], w.max_coords[1]]}.union([-1, 1])))
        hole_cells_x = np.array(
            [
                np.logical_and(cells_x[:-1] >= w.min_coords[0], cells_x[:-1] < w.max_coords[0])
                for w in holes
            ]).reshape((len(holes), len(cells_x) - 1))
        hole_cells_y = np.array(
            [
                np.logical_and(cells_y[:-1] >= w.min_coords[1], cells_y[:-1] < w.max_coords[1])
                for w in holes
            ]).reshape((len(holes), len(cells_y) - 1))

        cell_not_occupied = np.any(
            np.logical_and(hole_cells_x[:, np.newaxis, :], hole_cells_y[:, :, np.newaxis]), axis=0)
        cell_occupied = np.logical_not(cell_not_occupied)

        output_rects = []
        scale = self.table_top_accessible_extents / 2
        for y, co_arr in enumerate(cell_occupied):
            current_start_x = None
            prev_cell_occupied = False
            for x, current_cell_occupied in enumerate(chain(co_arr, [False])):
                if not current_cell_occupied and prev_cell_occupied:
                    min_coords = np.array([current_start_x, cells_y[y]]) * scale
                    max_coords = np.array([cells_x[x], cells_y[y + 1]]) * scale
                    rect = Rectangle(min_coords, max_coords)
                    output_rects.append(rect)
                if not prev_cell_occupied:
                    current_start_x = cells_x[x]
                prev_cell_occupied = current_cell_occupied
        return output_rects

    @property
    def target_zone_pos_table_frame(self) -> np.ndarray:
        return self.__target_zone_pos_table_frame

    @property
    def target_zone_extents(self) -> np.ndarray:
        return self.__target_zone_extents

    @property
    def ball(self) -> Optional[SimulationObject]:
        return self.__ball

    @property
    def table_extents(self) -> np.ndarray:
        return self.__table_extents

    @property
    def table_top_accessible_extents(self):
        return self.__table_extents[:2] - self.__barrier_width * 2

    @property
    def table_top_center_pose(self) -> Transformation:
        return self.__table_top_center

    @property
    def gripper_distance_to_table(self) -> float:
        return self.__gripper_distance_to_table

    @property
    def barrier_width(self) -> float:
        return self.__barrier_width

    @property
    def ball_radius(self) -> float:
        return self.__ball_radius
