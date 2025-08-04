import logging
import re
import types  # don't remove this, this is required for monkey patching in `exec` function # noqa
from dataclasses import dataclass, field
from typing import List

import numpy as np

import src.scene.relationships as sg
import src.utils.misc as misc
from src.scene.scene_graph import SceneGraph
from src.type import Character, Event, Object


@dataclass
class Area:
    conditions: List[str] = field(default_factory=list)


@dataclass
class Character_:
    name: str
    code_executor: "CodeExecutor"

    def set_position(self, position: Area):
        if self.name not in self.code_executor.parsed_event:
            self.code_executor.parsed_event[self.name] = {
                "position": [],
                "target_action": "",
            }

        if isinstance(position, (list, tuple)):  # wo_lcps
            self.code_executor.parsed_event[self.name]["position"] = position
        elif isinstance(position, Area):
            self.code_executor.parsed_event[self.name]["position"] = position.conditions
        else:
            raise ValueError(f"Invalid position type: {type(position)}")

    def set_orientation(self, target: str):
        if self.name not in self.code_executor.parsed_event:
            self.code_executor.parsed_event[self.name] = {
                "position": [],
                "target_action": "",
            }
        self.code_executor.parsed_event[self.name]["orientation"] = [f"look at {target}"]

    def set_target_action(self, action: str):
        if self.name not in self.code_executor.parsed_event:
            self.code_executor.parsed_event[self.name] = {
                "position": [],
                "target_action": "",
            }
        self.code_executor.parsed_event[self.name]["target_action"] = action


class CodeExecutor:
    def __init__(self, characters: List[Character], objects: List[Object], cfg):
        self.cfg = cfg
        self.characters = characters
        self.objects = objects
        self.scene_graph = SceneGraph("", cfg, objects)
        self.finder = self.scene_graph._obj_finder
        self.is_previous_event_set = False
        self.previous_events: List[Event] = []
        self.last_plan = {}
        self.logger = misc.get_console_logger("CodeExecutor", level=cfg.logging_level)

        # for dynamic code execution
        self.parsed_event = {}
        self.replacement_rules = {
            "def parse_event():": "def parse_event(self):",  # for monkey patching
            "def generate_next_plan():": "def parse_event(self):",  # for monkey patching, when code comes from Narrator
        }

        # add instance methods starting with "_get_" to replacement rules
        # e.g. get_objects_supported_by -> self._get_objects_supported_by
        instance_methods = [func for func in dir(self) if callable(getattr(self, func))]
        instance_methods = [func for func in instance_methods if not func.startswith("__")]
        for method in instance_methods:
            if re.match(r"_get_.*", method) or re.match(r"_is_.*", method):
                self.replacement_rules[f"{method[1:]}"] = f"self.{method}"

        self.pattern = re.compile(
            "|".join(
                map(
                    re.escape,
                    sorted(self.replacement_rules.keys(), key=len, reverse=True),
                )
            )
        )

    def set_previous_events(self, previous_events: List[Event]):
        def get_main_action_label(action_labels):
            if not action_labels:
                return ""
            if len(action_labels) == 1:
                return action_labels[0]
            return next(
                action for action in action_labels if not action.startswith("sit") and not action.startswith("lie")
            )

        self.previous_events = previous_events
        self.last_plan = {}
        for event in self.previous_events:
            parsed_event = event.parsed_event
            for character, plan in parsed_event.items():
                self.last_plan[character] = {
                    "relationships": plan["relationships"],
                    "target_action": get_main_action_label(plan["target_action"]),
                    "state": event.state,
                }
        self.is_previous_event_set = True

    def execute_response(self, response):
        """
        Naive use of `exec` causes scope error when LLM uses `self` in the list comprehension.
        So, we first monkey patch the method and then execute.
        """
        # pattern to extract the code block
        match = re.search(r"```python\s*\n([\s\S]+?)\n\s*```", response)
        if match:
            response = match.group(1)

        # apply replacement rules
        response = self.pattern.sub(lambda match: self.replacement_rules[match.group(0)], response).strip()

        # reset the parsed event
        self.parsed_event = {}

        # context for dynamic code execution
        global_context = globals()
        local_context = locals()
        local_context.update({"self": self})

        # define `parse_event(self)` method
        exec(response, global_context, local_context)

        # monkey patch `parse_event(self)` method
        exec(
            "self.parse_event = types.MethodType(parse_event, self)",
            global_context,
            local_context,
        )

        # execute `parse_event(self)` method
        exec("self.parse_event()", global_context, local_context)

        # during execution, `self.parsed_event` will be updated by the methods:
        # character.set_position, character.set_orientation, character.set_target_action
        return self.parsed_event

    ###########################################################################################
    ######################### functions that will be used by LLM ##############################
    ###########################################################################################
    def _get_character(self, name):
        self.logger.debug(f"Getting character {name}")
        return Character_(name, self)

    def _get_intersected_area(self, area_1: Area, area_2: Area) -> Area:
        self.logger.debug(f"Getting intersected area between {area_1} and {area_2}")
        return Area(area_1.conditions + area_2.conditions)

    def _get_object_supporting(self, anchor: str):
        assert anchor in self.objects, f"{anchor} is not an object in the scene"
        self.logger.debug(f"Getting object supporting {anchor}")
        relationships = self.scene_graph.get_relationships(anchor)
        for rel in relationships:
            if rel[2] == "on":
                self.logger.debug(f"Object supporting {anchor}: {rel[1]}")
                return rel[1]
        self.logger.debug(f"No object supporting {anchor}")
        return None

    def _get_objects_supported_by(self, anchor: str, label: str = None) -> List[str]:
        assert anchor in self.objects, f"{anchor} is not an object in the scene"
        self.logger.debug(f"Getting object supported by {anchor}")
        ret = []
        for obj in self.objects:
            if label and obj.label not in (label, label[:-1]):  # for plural
                continue
            if obj == anchor:
                continue
            relationships = self.scene_graph.get_relationships(obj)
            for rel in relationships:
                if rel[2] == "on" and rel[1] == anchor:
                    ret.append(str(obj))
        self.logger.debug(f"Objects supported by {anchor}: {ret}")
        return ret

    def _get_objects_in_front_of(self, anchor: str, label: str = None) -> List[str]:
        assert anchor in self.objects, f"{anchor} is not an object in the scene"
        self.logger.debug(f"Getting objects in front of {anchor}")
        ret = []
        for obj in self.objects:
            if label and obj.label not in (label, label[:-1]):  # for plural
                continue

            relationships = self.scene_graph.get_relationships(obj)
            for rel in relationships:
                if rel[2] == "in front of" and rel[1] == anchor:
                    ret.append(str(obj))
        self.logger.debug(f"Objects in front of {anchor}: {ret}")
        return ret

    def _get_objects_behind(self, anchor: str, label: str = None) -> List[str]:
        self.logger.debug(f"Getting objects behind {anchor}")
        ret = []
        for obj in self.objects:
            if label and obj.label not in (label, label[:-1]):  # for plural
                continue

            relationships = self.scene_graph.get_relationships(obj)
            for rel in relationships:
                if rel[2] == "behind" and rel[1] == anchor:
                    ret.append(str(obj))
        self.logger.debug(f"Objects behind {anchor}: {ret}")
        return ret

    def _get_objects_left_of(self, anchor: str, label: str = None) -> List[str]:
        self.logger.debug(f"Getting objects left of {anchor}")
        ret = []
        for obj in self.objects:
            if label and obj.label not in (label, label[:-1]):  # for plural
                continue

            relationships = self.scene_graph.get_relationships(obj)
            for rel in relationships:
                if rel[2] == "to the left of" and rel[1] == anchor:
                    ret.append(str(obj))
        self.logger.debug(f"Objects left of {anchor}: {ret}")
        return ret

    def _get_objects_right_of(self, anchor: str, label: str = None) -> List[str]:
        self.logger.debug(f"Getting objects right of {anchor}")
        ret = []
        for obj in self.objects:
            if label and obj.label not in (label, label[:-1]):  # for plural
                continue

            relationships = self.scene_graph.get_relationships(obj)
            for rel in relationships:
                if rel[2] == "to the right of" and rel[1] == anchor:
                    ret.append(str(obj))
        self.logger.debug(f"Objects right of {anchor}: {ret}")
        return ret

    def _get_objects_between(self, anchor_1: str, anchor_2: str, label: str = None) -> List[str]:
        self.logger.debug(f"Getting objects between {anchor_1} and {anchor_2}")
        anchor_1 = self.scene_graph._obj_finder[anchor_1]
        anchor_2 = self.scene_graph._obj_finder[anchor_2]
        p1 = anchor_1.center[:2]
        p2 = anchor_2.center[:2]

        v = p2 - p1
        len_v = np.linalg.norm(v)

        ret = []
        for obj in self.objects:
            if label and obj.label not in (label, label[:-1]):
                continue

            p3 = obj.center[:2]
            w = p3 - p1
            t = np.dot(v, w) / (len_v * len_v + 1e-12)
            if not (0 <= t <= 1):
                continue

            p3_proj = p1 + t * v
            len_proj = np.linalg.norm(p3 - p3_proj)
            if len_proj < self.between_threshold:
                ret.append(str(obj))
        self.logger.debug(f"Objects between {anchor_1} and {anchor_2}: {ret}")
        return ret

    def _get_objects_close_to(self, anchor: str, label: str = None) -> List[str]:
        self.logger.debug(f"Getting objects close to {anchor}")
        ret = []
        relationships = self.scene_graph.get_relationships(anchor)
        for rel in relationships:
            if label and rel[1].label not in (label, label[:-1]):  # for plural
                continue
            if rel[2] in ("close to", "adjacent to"):
                ret.append(str(rel[1]))
        self.logger.debug(f"Objects close to {anchor}: {ret}")
        return ret

    def _get_objects_adjacent_to(self, anchor: str, label: str = None) -> List[str]:
        self.logger.debug(f"Getting objects adjacent to {anchor}")
        ret = []
        relationships = self.scene_graph.get_relationships(anchor)
        for rel in relationships:
            if label and rel[1].label not in (label, label[:-1]):  # for plural
                continue
            if rel[2] == "adjacent to":
                ret.append(str(rel[1]))
        self.logger.debug(f"Objects adjacent to {anchor}: {ret}")
        return ret

    def _get_objects_associated_with(self, anchor: str, label: str = None) -> List[str]:
        self.logger.debug(f"Getting objects associated with {anchor}")
        logger_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(logging.WARN)

        ret = []
        anchor_label = self.finder[anchor].label
        close_objects = self._get_objects_close_to(anchor, label=label)
        for obj in close_objects:
            if anchor == self._get_closest_object(obj, label=anchor_label):
                ret.append(obj)

        self.logger.setLevel(logger_level)
        self.logger.debug(f"Objects associated with {anchor}: {ret}")
        return ret

    def _get_closest_object(self, anchor: str, label: str = None) -> str:
        self.logger.debug(f"Getting closest object to {anchor}")
        anchor = self.finder[anchor]
        closest_obj = None
        min_dist = float("inf")

        for obj in self.objects:
            if label and obj.label not in (label, label[:-1]):
                continue
            if obj == anchor:
                continue

            dist = sg.get_distance_between_surfaces(anchor, obj)
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        self.logger.debug(f"Closest object to {anchor} with label {label}: {closest_obj}")
        return str(closest_obj)

    def _get_area_in_front_of(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area in front of {anchor}")
        return Area([f"in front of {anchor}"])

    def _get_area_behind(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area behind {anchor}")
        return Area([f"behind {anchor}"])

    def _get_area_left_of(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area to the left of {anchor}")
        return Area([f"to the left of {anchor}"])

    def _get_area_right_of(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area to the right of {anchor}")
        return Area([f"to the right of {anchor}"])

    def _get_area_close_to(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area close to {anchor}")
        return Area([f"close to {anchor}"])

    def _get_area_adjacent_to(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area adjacent to {anchor}")
        return Area([f"adjacent to {anchor}"])

    def _get_area_to_sit_on(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area to sit on {anchor}")
        return Area([f"sit on {anchor}"])

    def _get_area_to_interact_with(self, anchor: str) -> Area:
        self.logger.debug(f"Getting area to use {anchor}")
        return Area([f"adjacent to {anchor}"])

    def _get_area_between(self, anchor_1: str, anchor_2: str) -> Area:
        self.logger.debug(f"Getting area between {anchor_1} and {anchor_2}")
        return Area([f"between {anchor_1} and {anchor_2}"])

    def _get_area_aligned_with(self, anchor_1: str, anchor_2: str) -> Area:
        self.logger.debug(f"Getting area aligned with {anchor_1} and {anchor_2}")
        return Area([f"aligned with {anchor_1} and {anchor_2}"])

    def _get_distance_between(self, anchor_1: str, anchor_2: str) -> float:
        self.logger.debug(f"Getting distance between {anchor_1} and {anchor_2}")
        anchor_1 = self.finder[anchor_1]
        anchor_2 = self.finder[anchor_2]
        distance = sg.get_distance_between_surfaces(anchor_1, anchor_2)
        self.logger.debug(f"Distance between {anchor_1} and {anchor_2}: {distance}")
        return distance

    def _is_object_occupied(self, obj: str) -> bool:
        self.logger.debug(f"Checking if {obj} is occupied")
        assert self.is_previous_event_set, "Previous events are not set"

        # check if the object is occupied by someone who is sitting or lying on it
        for character, plan in self.last_plan.items():
            for rel in plan["relationships"]:
                if not (rel.startswith("sit on") or rel.startswith("lie on")):
                    continue

                anchor = rel.split()[-1]
                if anchor == obj:
                    self.logger.debug(f"State of {obj}: occupied")
                    return True

        # check if the object is occupied by someone who is using it
        for character, plan in self.last_plan.items():
            if plan["state"] == "completed":
                continue

            for rel in plan["relationships"]:
                if not (rel.startswith("adjacent to") or rel.startswith("close to")):
                    continue

                anchor = rel.split()[-1]
                anchor = self.finder[anchor]
                if anchor != obj:
                    continue
                for action in plan["target_action"]:
                    if action.startswith("use") and anchor.label in action:
                        self.logger.debug(f"State of {obj}: occupied")
                        return True

        self.logger.debug(f"State of {obj}: unoccupied")
        return False

    def _is_object_of_label(self, obj: str, label: str) -> bool:
        self.logger.debug(f"Checking if {obj} is of label '{label}'")
        if label == self.finder[obj].label:
            self.logger.debug(f"{obj} is of label '{label}'")
            return True

        self.logger.debug(f"{obj} is not of label '{label}'")
        return False
