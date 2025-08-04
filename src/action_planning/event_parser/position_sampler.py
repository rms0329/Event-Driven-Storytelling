import random
import re
from typing import List

import numpy as np
import torch

import src.scene.relationships as sg
from src.scene.grid_map import GridMap
from src.type import Character, Object
from src.utils import misc


class PositionSampler:
    def __init__(self, grid_map: GridMap, cfg) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.grid_map = grid_map

        self.distance_threshold = cfg.position_sampler.distance_threshold
        self.multi_thresholds = cfg.position_sampler.multi_object_thresholds
        self.avoidance_threshold = cfg.position_sampler.avoidance_threshold
        self.avoidance_coeff = cfg.position_sampler.avoidance_coefficient
        self.nearby_threshold = cfg.position_sampler.nearby_threshold
        self.init_temperature = cfg.position_sampler.init_temperature
        self.final_temperature = cfg.position_sampler.final_temperature
        self.cooling_rate = cfg.position_sampler.cooling_rate
        self.cooling_interval = cfg.position_sampler.cooling_interval
        self.move_std = cfg.position_sampler.move_std
        self.convergence_threshold = cfg.position_sampler.convergence_threshold
        self.convergence_iteration = cfg.position_sampler.convergence_iteration
        self.w_directional = cfg.position_sampler.cost_weights.directional
        self.w_distance = cfg.position_sampler.cost_weights.distance
        self.w_multi = cfg.position_sampler.cost_weights.multi_object
        self.w_interaction = cfg.position_sampler.cost_weights.interaction
        self.w_avoidance = cfg.position_sampler.cost_weights.avoidance
        self.w_nearby = cfg.position_sampler.cost_weights.nearby
        self.k = cfg.position_sampler.k
        self.r = cfg.position_sampler.r
        self.max_iter = cfg.position_sampler.max_iter
        self.collision_check_type = cfg.position_sampler.collision_check_type
        self.ground_support_threshold = cfg.scene_graph.boundary_offset
        self.floor_level = 0.0
        assert self.collision_check_type in ["cylinder", "bbox"]

        # patterns for different types of conditions
        self.multi_obj_pattern = ["between", "aligned with"]
        self.multi_obj_pattern = r"^(" + "|".join(self.multi_obj_pattern) + ")"
        self.distance_pattern = list(self.distance_threshold.keys())
        self.distance_pattern = r"^(" + "|".join(self.distance_pattern) + ")"
        self.interaction_pattern = ["sit on", "lie on"]
        self.interaction_pattern = r"^(" + "|".join(self.interaction_pattern) + ")"
        self.directional_pattern = [
            "to the left of",
            "to the right of",
            "in front of",
            "behind",
        ]
        self.directional_pattern = r"^(" + "|".join(self.directional_pattern) + ")"
        self.orientation_pattern = ["look at"]
        self.orientation_pattern = r"^(" + "|".join(self.orientation_pattern) + ")"

    def update_anchors(self, characters: List[Character], objects: List[Object]):
        self.objects = objects
        self.anchor_finder = {f"{obj.label}_{obj.idx}": obj for obj in objects}
        self.anchor_finder.update({character.name: character for character in characters})

    def sample_position(
        self,
        conditions: List[str],
        initial_position: np.ndarray = None,
        position_to_nearby: np.ndarray = None,
        positions_to_avoid: List[np.ndarray] = None,
        return_tensor: bool = True,
        verbose: bool = False,
    ):
        if self.is_deterministic(conditions):  # lie on [any] or sit on [chair]
            position = self.sample_deterministic_position(conditions)
            if return_tensor:
                return torch.tensor(position, dtype=torch.float32, device=self.device)
            return position

        position_to_nearby, positions_to_avoid = self._preprocess(position_to_nearby, positions_to_avoid)
        if initial_position is None:
            initial_position = self._get_initial_position(conditions)

        # Initial parameters
        current_temp = self.init_temperature
        min_temp = self.final_temperature
        cooling_rate = self.cooling_rate

        current_position = initial_position  # (2,)
        ignore_collision = any("sit on" in condition or "lie on" in condition for condition in conditions)
        current_cost = self.calculate_cost(
            current_position,
            conditions,
            position_to_nearby,
            positions_to_avoid,
            ignore_collision,
        )

        step_count = 0
        update_count = 0
        converge_count = 0
        while converge_count < self.convergence_iteration:
            # Generate a new candidate solution
            new_position = current_position + np.random.normal(0, self.move_std, 2).astype(np.float32)
            new_cost = self.calculate_cost(
                new_position,
                conditions,
                position_to_nearby,
                positions_to_avoid,
                ignore_collision,
            )

            # Decide if we should accept the new solution
            if self.acceptance_probability(current_cost, new_cost, current_temp) > random.random():
                change_rate = abs(new_cost - current_cost) / (current_cost + 1e-6)
                current_position = new_position
                current_cost = new_cost

                converge_count = converge_count + 1 if change_rate < self.convergence_threshold else 0
                update_count += 1
                if update_count % self.cooling_interval == 0:
                    current_temp *= cooling_rate
                    current_temp = max(current_temp, min_temp)

            step_count += 1
            if step_count >= self.max_iter:
                break

        if verbose:
            print(f"Step count: {step_count}")
            print(f"Update count: {update_count}")

        current_position = misc.add_z(current_position, z=0).reshape(1, 3)
        if return_tensor:
            return torch.tensor(current_position, dtype=torch.float32, device=self.device)
        return current_position

    def sample_deterministic_position(self, conditions):
        for condition in conditions:
            if ("lie on" in condition) or ("sit on" in condition):
                anchor = condition.split(" ")[-1]
                anchor = self.anchor_finder[anchor]
                return misc.add_z(
                    anchor.center[:2] + anchor.orientation[:2] * (anchor.depth * anchor.sit_offset_horizontal),
                    z=0,
                ).reshape(1, 3)

        assert False, "Should not reach here"

    def is_deterministic(self, conditions):
        for condition in conditions:
            if "lie on" in condition:
                return True
            if "sit on" in condition:
                anchor = condition.split(" ")[-1]
                anchor = self.anchor_finder[anchor]
                if anchor.label in ["chair", "armchair", "stool"]:
                    return True
        return False

    def acceptance_probability(self, old_cost, new_cost, temperature):
        # If the new solution is better, accept it
        if new_cost < old_cost:
            return 1.0
        # If the new solution is worse, calculate an acceptance probability
        return np.exp((old_cost - new_cost) / temperature)

    def calculate_cost(
        self,
        position,
        conditions,
        position_to_nearby=None,
        positions_to_avoid=None,
        ignore_collision=False,
    ):
        if self.grid_map.is_off_boundary(position):
            return np.inf

        cost = 0.0
        if not ignore_collision and self.is_penetration_occured(position):
            cost += 100.0
        for condition in conditions:
            if re.match(self.multi_obj_pattern, condition):
                cost += self.w_multi * self.multi_object_cost(position, condition)
            elif re.match(self.distance_pattern, condition):
                cost += self.w_distance * self.distance_cost(position, condition)
            elif re.match(self.directional_pattern, condition):
                cost += self.w_directional * self.directional_cost(position, condition)
            elif re.match(self.interaction_pattern, condition):
                cost += self.w_interaction * self.interaction_cost(position, condition)
            elif re.match(self.orientation_pattern, condition):
                pass
            else:
                raise ValueError(f"Invalid condition: {condition}")

        if position_to_nearby is not None:
            cost += self.w_nearby * self.nearby_cost(position, position_to_nearby)
        if len(positions_to_avoid) > 0:
            cost += self.w_avoidance * self.avoidance_cost(position, positions_to_avoid)
        return cost

    def is_penetration_occured(self, position, offset=0.0):
        if self.collision_check_type == "cylinder":
            return self.grid_map.is_penetration_occured(position)
        assert position.ndim == 1 and position.shape[0] == 2, "position must be a 2D vector"

        for obj in self.objects:
            if abs(obj.min_z - self.floor_level) > self.ground_support_threshold:
                continue

            obj_center = obj.center[:2]
            obj_orientation = obj.orientation
            obj_width = obj.width
            obj_depth = obj.depth

            x, y = misc.canonicalize_situation(position, obj_center, obj_orientation)
            max_x = obj_width / 2 - offset
            min_x = -obj_width / 2 + offset
            max_y = obj_depth / 2 - offset
            min_y = -obj_depth / 2 + offset
            if not (x <= min_x or x >= max_x or y <= min_y or y >= max_y):
                return True

        return False

    def directional_cost(self, position, condition):
        predicate = " ".join(condition.split(" ")[:-1])
        if predicate not in [
            "to the left of",
            "to the right of",
            "in front of",
            "behind",
        ]:
            return 0.0

        anchor = condition.split(" ")[-1]
        anchor = self.anchor_finder[anchor]
        if isinstance(anchor, Character):
            return 0.0
        if not anchor.has_orientation:
            return 0.0

        distance = sg.get_distance_from_relationship(position, anchor, predicate)
        if distance == 0.0:
            return 0.0
        return self.k * np.exp(distance / self.r)

    def distance_cost(self, position, condition):
        predicate = " ".join(condition.split(" ")[:-1])
        anchor = condition.split(" ")[-1]
        if predicate not in self.distance_threshold:
            return 0.0

        anchor = condition.split(" ")[-1]
        anchor = self.anchor_finder[anchor]
        if isinstance(anchor, Character):
            return 0.0

        distance = sg.get_distance_from_boundary(position, anchor)
        threshold = self.distance_threshold[predicate]
        if distance < threshold:
            return 0.0
        return self.k * np.exp((distance - threshold) / self.r)

    def interaction_cost(self, position, condition):
        predicate = " ".join(condition.split(" ")[:-1])
        if predicate not in ["sit on", "lie on"]:
            return 0.0

        anchor = condition.split(" ")[-1]
        anchor = self.anchor_finder[anchor]
        if isinstance(anchor, Character):
            return 0.0

        distance = sg.get_distance_from_interaction(position, anchor, predicate)
        return self.k * np.exp(distance / self.r)

    def multi_object_cost(self, position, condition):
        words = condition.split(" ")
        anchor_1 = words[-3]
        anchor_1 = self.anchor_finder[anchor_1]
        anchor_2 = words[-1]
        anchor_2 = self.anchor_finder[anchor_2]
        if isinstance(anchor_1, Character) or isinstance(anchor_2, Character):
            return 0.0

        predicate = " ".join(words[:-3])
        distance = sg.get_distance_from_multi_obj_relationship(
            position,
            anchor_1,
            anchor_2,
            predicate,
            threshold_aligned=self.multi_thresholds[predicate]["aligned"],
            threshold_orthogonal=self.multi_thresholds[predicate]["orthogonal"],
        )
        return self.k * np.exp(distance / self.r)

    def nearby_cost(self, position, position_to_nearby):
        distance = np.linalg.norm(position - position_to_nearby)
        if distance < self.nearby_threshold:
            return 0.0
        return self.k * np.exp((distance - self.nearby_threshold) / self.r)

    def avoidance_cost(self, position, positions_to_avoid):
        cost = 0
        for avoidance_position in positions_to_avoid:
            distance = np.linalg.norm(position - avoidance_position)
            if distance > self.avoidance_threshold:
                continue
            cost += self.k * np.exp(self.avoidance_coeff * (self.avoidance_threshold - distance) / self.r)
        return cost

    def _preprocess(self, position_to_nearby, positions_to_avoid):
        if position_to_nearby is not None:
            if isinstance(position_to_nearby, torch.Tensor):
                position_to_nearby = position_to_nearby.cpu().numpy()
            if position_to_nearby.shape != (2,):
                position_to_nearby = position_to_nearby.squeeze()[:2]

        if positions_to_avoid is None:
            positions_to_avoid = []

        if len(positions_to_avoid) > 0:
            if isinstance(positions_to_avoid[0], torch.Tensor):
                positions_to_avoid = [p.cpu().numpy() for p in positions_to_avoid]
            if positions_to_avoid[0].shape != (2,):
                positions_to_avoid = [p.squeeze()[:2] for p in positions_to_avoid]
        return position_to_nearby, positions_to_avoid

    def _get_initial_position(self, conditions):
        for condition in conditions:
            if re.match(self.multi_obj_pattern, condition):
                anchor_1 = condition.split(" ")[-3]
                anchor_1 = self.anchor_finder[anchor_1]
                anchor_2 = condition.split(" ")[-1]
                anchor_2 = self.anchor_finder[anchor_2]
                if isinstance(anchor_1, Object) and isinstance(anchor_2, Object):
                    return (anchor_1.center[:2] + anchor_2.center[:2]) / 2

            if (
                re.match(self.distance_pattern, condition)
                or re.match(self.interaction_pattern, condition)
                or re.match(self.directional_pattern, condition)
            ):
                anchor = condition.split(" ")[-1]
                anchor = self.anchor_finder[anchor]
                if isinstance(anchor, Object):
                    return anchor.center[:2]

        position = self.grid_map.get_random_position()
        position = position[:2]
        return position
