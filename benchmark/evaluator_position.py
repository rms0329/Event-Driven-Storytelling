import re
from typing import List

import numpy as np

from src.scene import relationships as sg
from src.type import Object


class PositionEvaluator:
    def __init__(self) -> None:
        self.objects = []
        self.finder = {}
        self.thresholds = {
            "to the left of": 0.1,
            "to the right of": 0.1,
            "in front of": 0.1,
            "behind": 0.1,
            "adjacent to": 0.7,
            "close to": 1.4,
            "inside": 0.1,
        }
        self.ground_support_threshold = 0.1
        self.floor_level = 0.0

    def reset_objects(self, objects: List[Object]):
        self.objects = objects
        self.finder = {f"{obj.label}_{obj.idx}": obj for obj in objects}

    def evaluate(self, condition, position):
        assert self.objects, "objects must be set before evaluation"
        if isinstance(condition, (list, np.ndarray)):
            return self._check_by_area(condition, position)

        if not condition:
            return True, 0.0
        position = np.array(position, dtype=np.float32)
        tokens = re.split(r"(\|\||&&|\(|\))", condition)
        tokens = [t.strip() for t in tokens if t.strip() != ""]
        value, error, index = self._parse_expression(tokens, 0, position)
        if index != len(tokens):
            raise SyntaxError("Unexpected tokens at the end")
        return value, error

    def _parse_expression(self, tokens, index, position):
        value, error, index = self._parse_term(tokens, index, position)
        while index < len(tokens) and tokens[index] == "||":
            index += 1  # Skip '||'
            right_value, right_error, index = self._parse_term(tokens, index, position)
            value = value or right_value
            error = min(error, right_error)
        return value, error, index

    def _parse_term(self, tokens, index, position):
        value, error, index = self._parse_factor(tokens, index, position)
        while index < len(tokens) and tokens[index] == "&&":
            index += 1  # Skip '&&'
            right_value, right_error, index = self._parse_factor(tokens, index, position)
            value = value and right_value
            error = max(error, right_error)
        return value, error, index

    def _parse_factor(self, tokens, index, position):
        if tokens[index] == "(":
            index += 1  # Skip '('
            value, error, index = self._parse_expression(tokens, index, position)
            if index >= len(tokens) or tokens[index] != ")":
                raise SyntaxError("Expected ')'")
            index += 1  # Skip ')'
            return value, error, index
        else:
            value, error = self._check(tokens[index], position)
            index += 1
            return value, error, index

    def _check(self, condition_str, position):
        anchor = condition_str.split(" ")[-1]
        anchor = self.finder[anchor]
        predicate = " ".join(condition_str.split(" ")[:-1])
        if predicate in ["to the left of", "to the right of", "in front of", "behind"]:
            distance = sg.get_distance_from_relationship(position, anchor, predicate, offset=0.0)
        elif predicate in ["adjacent to", "close to", "inside"]:
            distance = sg.get_distance_from_boundary(position, anchor, offset=0.0)
        else:
            raise ValueError(f"Unknown predicate: {predicate}")

        passed = distance <= self.thresholds[predicate]
        error = max(0, distance - self.thresholds[predicate])
        return passed, error

    def _check_by_area(self, expected_area, position):
        position = np.array(position, dtype=np.float32)
        expected_area = np.array(expected_area, dtype=np.float32)
        min_x, min_y, max_x, max_y = expected_area
        x, y = position
        if (min_x <= x <= max_x) and (min_y <= y <= max_y):
            return True, 0.0

        # when the position is outside the area
        if x <= min_x:
            if y <= min_y:
                error = np.linalg.norm([x - min_x, y - min_y])
            elif y >= max_y:
                error = np.linalg.norm([x - min_x, y - max_y])
            else:
                error = min_x - x
        elif x >= max_x:
            if y <= min_y:
                error = np.linalg.norm([x - max_x, y - min_y])
            elif y >= max_y:
                error = np.linalg.norm([x - max_x, y - max_y])
            else:
                error = x - max_x
        else:
            if y <= min_y:
                error = min_y - y
            else:
                error = y - max_y

        assert error >= 0, f"error must be non-negative: {error}"
        return False, float(error)

    def check_collision(self, position, offset=0.0):
        position = np.array(position, dtype=np.float32)
        assert position.ndim == 1 and position.shape[0] == 2, "position must be a 2D vector"

        for obj in self.objects:
            if abs(obj.min_z - self.floor_level) > self.ground_support_threshold:
                continue

            obj_center = obj.center[:2]
            obj_orientation = obj.orientation
            obj_width = obj.width
            obj_depth = obj.depth

            x, y = sg.canonicalize_situation(position, obj_center, obj_orientation)
            max_x = obj_width / 2 - offset
            min_x = -obj_width / 2 + offset
            max_y = obj_depth / 2 - offset
            min_y = -obj_depth / 2 + offset
            if x <= min_x or x >= max_x or y <= min_y or y >= max_y:
                continue

            collision_distance = min(
                abs(x - min_x),
                abs(x - max_x),
                abs(y - min_y),
                abs(y - max_y),
            )
            return True, collision_distance
        return False, 0.0
