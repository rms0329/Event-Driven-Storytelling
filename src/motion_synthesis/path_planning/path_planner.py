from collections import defaultdict
from typing import List

import numpy as np
import torch

from src.scene.grid_map import GridMap
from src.type import Character
from src.utils import misc, smpl
from src.utils.transform import quaternion_apply

from ..states import State
from ..utils import spline, spring
from .priority_queue import PriorityQueue


class PathPlanner:
    def __init__(self, grid_map: GridMap, cfg) -> None:
        self.cfg = cfg
        self.grid_map = grid_map
        self.reserved_paths = []

        self.device = cfg.device
        self.framerate = cfg.framerate
        self.dt = 1 / self.framerate
        self.min_planning_timesteps = cfg.path_planner.min_planning_timesteps
        self.max_planning_timesteps = cfg.path_planner.max_planning_timesteps
        self.oracle_timesteps = cfg.path_planner.oracle_timesteps
        self.non_moving_cost = cfg.path_planner.non_moving_cost
        self.collision_soft_threshold = cfg.path_planner.collision_soft_threshold
        self.collision_hard_threshold = cfg.path_planner.collision_hard_threshold
        self.collision_cost_scaler = cfg.path_planner.collision_cost_scaler
        self.soft_collision_cost = cfg.path_planner.soft_collision_cost
        self.hard_collision_cost = cfg.path_planner.hard_collision_cost
        self.future_offsets = torch.tensor(
            [int(offset * self.framerate) for offset in cfg.motion_matching.future_offsets],
            device=self.device,
        )
        self.far_threshold = cfg.state_manager.far_threshold
        self.x_axis = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).reshape(1, 3)

    def update_character_path(self, character: Character, characters: List[Character]):
        existing_paths = []
        occupied_positions = []
        for other in characters:
            if character == other:
                continue

            if other == character.interactee:
                if other.is_resting_posture():
                    continue  # don't consider stationary interactee's path
                else:
                    existing_paths.append(other.grid_path)  # do not push away interactee
            else:
                if (
                    other.state in [State.INTERACTING, State.TRANSITION_IN, State.TRANSITION_OUT]
                    or other.is_resting_posture()
                ):
                    occupied_position = other.position
                    occupied_position = self.grid_map.get_closest_grid_position(occupied_position)
                    occupied_positions.append(occupied_position)
                elif other.state == State.IDLE:
                    continue
                else:
                    existing_paths.append(other.grid_path)

        # update the reserved path (grid level)
        if character.is_resting_posture():
            character.grid_path = [
                self.grid_map.get_closest_grid_position(character.position, exclude_occupied=False)
            ] * self.min_planning_timesteps
        else:
            character.grid_path = self.search_path(
                start=self.grid_map.get_closest_grid_position(character.position),
                goal=self.grid_map.get_closest_grid_position(character.target_position),
                min_timesteps=self.min_planning_timesteps,
                max_timesteps=self.max_planning_timesteps,
                oracle_timesteps=self.oracle_timesteps,
                occupied_positions=occupied_positions,
                existing_paths=existing_paths,
            )

    def search_path(
        self,
        start,
        goal,
        start_time=0,
        min_timesteps=0,
        max_timesteps=100,
        existing_paths=None,
        occupied_positions=None,
        oracle_timesteps=20,
    ):
        priority_queue = PriorityQueue((start_time, start), priority=0)
        came_from = {(start_time, start): (-1, None)}
        lowest_cost = defaultdict(lambda: float("inf"))
        lowest_cost[start_time, start] = 0

        while not priority_queue.is_empty():
            current_time, current = priority_queue.pop()
            if current == goal and current_time >= start_time + min_timesteps:
                return self._reconstruct_path(came_from, goal, current_time)

            neighbors = self._get_neighbors(current)
            if current_time >= start_time + max_timesteps:
                neighbors.append(goal)  # make a jump to the goal

            for next in neighbors:
                next_time = current_time + 1
                if not self._is_within_grid_map(next):
                    continue

                max_consider_time = start_time + oracle_timesteps
                new_cost, collision_cost = self._calculate_new_cost(
                    lowest_cost,
                    current,
                    next,
                    next_time,
                    existing_paths,
                    occupied_positions,
                    max_consider_time,
                )

                if new_cost < lowest_cost[next_time, next]:
                    came_from[next_time, next] = (current_time, current)
                    lowest_cost[next_time, next] = new_cost

                    priority = new_cost + self._heuristic(next, goal)
                    entry = (next_time, next)
                    if entry in priority_queue:
                        priority_queue.update(entry, priority=priority)
                    else:
                        priority_queue.push(entry, priority=priority)
        raise RuntimeError("There is no path that can reach the goal")

    def _get_neighbors(self, current):
        dxs = (0, 0, -1, -1, -1, 1, 1, 1, 0)
        dys = (1, -1, -1, 0, 1, -1, 0, 1, 0)

        neighbors = []
        for dx, dy in zip(dxs, dys):
            neighbors.append((current[0] + dx, current[1] + dy))
        return neighbors

    def _is_within_grid_map(self, position):
        return (0 <= position[0] < self.grid_map.num_x) and (0 <= position[1] < self.grid_map.num_y)

    def _calculate_new_cost(
        self,
        lowest_cost,
        current,
        next,
        next_time,
        existing_paths,
        occupied_positions,
        max_consider_time,
    ):
        current_time = next_time - 1
        result = lowest_cost[current_time, current] + self._get_moving_cost(current, next)
        if self.grid_map.occupied[next]:
            result += 1e5

        collision_cost = 0
        if existing_paths is not None and next_time <= max_consider_time:
            collision_cost = self._get_collision_cost(next, next_time, existing_paths, occupied_positions)
            result += collision_cost

        return result, collision_cost

    def _get_moving_cost(self, current, next):
        dx = next[0] - current[0]
        dy = next[1] - current[1]
        if dx == 0 and dy == 0:
            return self.non_moving_cost
        return 1 if dx == 0 or dy == 0 else (2**0.5)

    def _get_collision_cost(self, position, t, existing_paths, occupied_positions):
        cost = 0
        for path in existing_paths:
            if not path or t >= len(path):
                continue

            distance = np.linalg.norm(self.grid_map[position] - self.grid_map[path[t]])
            cost += self._collision_cost(distance)

        for occupied_position in occupied_positions:
            distance = np.linalg.norm(self.grid_map[position] - self.grid_map[occupied_position])
            cost += self._collision_cost(distance)

        return cost

    def _collision_cost(self, distance):
        if distance > self.collision_soft_threshold:
            return self.soft_collision_cost
        if distance < self.collision_hard_threshold:
            return self.hard_collision_cost

        scale = self.collision_cost_scaler / (self.collision_soft_threshold**2)
        return scale * ((self.collision_soft_threshold - distance) ** 2)

    def _heuristic(self, current, goal, D=1, D2=2**0.5):  # Diagonal Distance
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def _reconstruct_path(self, came_from, goal, arrival_time):
        route = []
        visited = set()
        current = goal
        current_time = arrival_time
        while current:
            if (current_time, current) in visited:
                raise RuntimeError("There is a loop in the path")
            route.append(current)
            visited.add((current_time, current))
            current_time, current = came_from[current_time, current]
        route = route[::-1]
        return route

    def update_future_prediction(self, character: Character):
        spline_curve, _ = self.get_tcb_spline_curve(character.grid_path, velocity_mps=character.velocity_mps)
        spline_curve = torch.tensor(spline_curve, device=self.device)
        character.spline_curve = spline_curve[: min(self.future_offsets)]

        ### update future positions
        future_positions = []
        for offset in self.future_offsets:
            offset = min(offset, len(spline_curve) - 1)
            future_positions.append(spline_curve[offset])
        character.future_positions = torch.stack(future_positions, dim=0)

        ### update future facing directions
        distance_to_target = torch.norm(character.position - character.target_position)
        # during approaching
        if distance_to_target >= self.far_threshold:
            target_direction = character.future_positions[-1:] - character.position
            target_direction /= torch.norm(target_direction, dim=-1, keepdim=True)
        # waiting interactee
        elif character.interactee is not None:
            direction_to_interactee = character.interactee.position - character.position
            direction_to_interactee /= torch.norm(direction_to_interactee, dim=-1, keepdim=True)
            if character.is_resting_posture():
                target_direction = misc.calculate_rotated_direction(
                    character.target_facing_direction,
                    direction_to_interactee,
                    threshold_angle=torch.pi / 4,
                )
            else:
                target_direction = direction_to_interactee
        # when target direction is given (e.g. human-scene interaction)
        elif character.target_facing_direction is not None:
            target_direction = character.target_facing_direction
        # default case
        else:
            target_direction = character.facing_direction

        target_rotation = smpl.get_character_rotations(target_direction)
        future_rotations, _ = spring.simple_spring_damper_exact_quaternion(
            character.rotation,
            character.angular_velocity,
            target_rotation,
            character.halflife_rotation,
            (self.future_offsets * self.dt).unsqueeze(-1),  # for broadcasting
        )
        character.future_facing_directions = quaternion_apply(future_rotations, self.x_axis)

    def get_tcb_spline_curve(self, grid_path, velocity_mps=0.9, tension=-1.0, continuity=0.0, bias=0.0):
        velocity_mpf = velocity_mps / self.framerate
        control_points = np.array([self.grid_map[p][:2] for p in grid_path]).reshape(1, -1, 2)
        control_points = np.concatenate([control_points, np.zeros((*control_points.shape[:-1], 1))], axis=-1)
        ks = np.empty((0,), dtype=np.int32)
        ts = np.empty((0,), dtype=np.float32)
        for i, (prev_grid, next_grid) in enumerate(zip(grid_path[:-1], grid_path[1:])):
            prev_pos = self.grid_map[prev_grid]
            next_pos = self.grid_map[next_grid]
            distance = np.linalg.norm(next_pos - prev_pos)
            num_frames = int(distance / velocity_mpf)
            ks = np.append(ks, np.repeat(i, num_frames))
            ts = np.append(ts, np.linspace(0, 1, num_frames, endpoint=False))
        ks = np.append(ks, i)
        ts = np.append(ts, 1.0)

        spline_curve = spline.interpolate_tcb_spline(
            control_points,
            ks,
            ts,
            tension=tension,
            continuity=continuity,
            bias=bias,
        )
        _, tangents = spline.calculate_tangents(spline_curve, tension=tension, continuity=continuity, bias=bias)

        # z-value should be set to 0.0
        return (
            misc.add_z(spline_curve[0, :, :2], z=0.0),
            misc.add_z(tangents[0, :, :2], z=0.0),
        )

    # def _update_future_prediction_spline(self, character: Character):
    #     if character.state == State.IDLE:
    #         self._update_future_prediction_idle(character)
    #         return

    #     spline_curve, tangents = self.get_tcb_spline_curve(character.grid_path, velocity_mps=character.velocity_mps)
    #     spline_curve = torch.tensor(spline_curve, device=self.device)
    #     tangents = torch.tensor(tangents, device=self.device)

    #     future_positions = []
    #     future_facing_directions = []
    #     for offset in self.future_offsets:
    #         offset = min(offset, len(spline_curve) - 1)
    #         future_positions.append(spline_curve[offset])
    #         future_facing_directions.append(F.normalize(tangents[offset], dim=-1))

    #     character.future_positions = torch.stack(future_positions, dim=0)
    #     character.future_facing_directions = torch.stack(future_facing_directions, dim=0)

    # def _update_future_prediction_spring(self, character: Character):
    #     if character.state == State.IDLE:
    #         self._update_future_prediction_idle(character)
    #         return

    #     current_position = torch.tensor(self.grid_map[character.grid_path[0]], device=self.device).reshape(1, 3)
    #     target_position = torch.tensor(self.grid_map[character.grid_path[1]], device=self.device).reshape(1, 3)
    #     target_direction = target_position - current_position
    #     target_direction /= torch.norm(target_direction, dim=-1, keepdim=True)

    #     target_velocity = target_direction * character.velocity_mps
    #     target_rotation = smpl.get_character_rotations(target_direction)

    #     character.future_positions, *_ = spring.spring_character_update(
    #         current_position,
    #         character.velocity,
    #         character.acceleration,
    #         target_velocity,
    #         character.halflife_position,
    #         (self.future_offsets * self.dt).unsqueeze(-1),  # for broadcasting
    #     )

    #     future_rotations, _ = spring.simple_spring_damper_exact_quaternion(
    #         character.rotation,
    #         character.angular_velocity,
    #         target_rotation,
    #         character.halflife_rotation,
    #         (self.future_offsets * self.dt).unsqueeze(-1),  # for broadcasting
    #     )
    #     character.future_facing_directions = quaternion_apply(future_rotations, self.x_axis)
