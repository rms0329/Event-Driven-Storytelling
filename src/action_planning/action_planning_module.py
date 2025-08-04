import queue
import re
from typing import List

import numpy as np
import torch

from ..motion_synthesis.state_manager import StateManager
from ..motion_synthesis.states import State
from ..scene.scene import Scene
from ..type import Character, Event, Object
from ..utils import misc
from .event_parser import get_event_parser
from .event_parser.event_parser import _VALID_RELATIONSHIPS
from .event_parser.position_sampler import PositionSampler
from .narrator import get_narrator
from .scene_describer.scene_describer import SceneDescriber


class ActionPlanningModule:
    def __init__(self, scene: Scene, characters: List[Character], state_manager: StateManager, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.scene = scene
        self.characters = characters
        self.state_manager = state_manager

        self.scene_describer = SceneDescriber(cfg)
        self.scene_description = self.scene_describer.get_scene_description(scene.scene_name, scene.scene_graph)
        self.narrator = get_narrator(characters, scene.objects, cfg)
        self.event_parser = get_event_parser(characters, scene.objects, cfg)
        self.event_queue = queue.Queue()
        self.position_sampler = PositionSampler(scene.grid_map, cfg)
        self.position_sampler.update_anchors(characters, scene.objects)

        self.finder = {f"{obj.label}_{obj.idx}": obj for obj in scene.objects}
        self.finder.update({character.name: character for character in self.characters})
        self.logger = misc.get_console_logger("ActionPlanningModule", cfg.logging_level)

        self.tags = self.state_manager.motion_matcher.motion_db.tags
        self.hhi_tags = self.state_manager.motion_matcher.motion_db.hhi_tags
        self.minimum_distance = cfg.misc.minimum_distance
        self.root_height_offset = cfg.misc.root_height_offset
        self.position_relationships_pattern = re.compile(
            r"^(" + "|".join(rel for rel in _VALID_RELATIONSHIPS if rel != "look at") + ")"
        )

    def is_new_event_required(self):
        return self.narrator.is_new_event_required()

    def generate_new_event(self, user_instruction=None):
        event, parsed_event = self._get_subsequent_event(user_instruction)
        if event is None:
            self.logger.info("Skipping the planning...")
            return
        self.event_queue.put((event, parsed_event))
        self.logger.info("Planning done. New event is created to the queue.")

    def _get_subsequent_event(self, user_instruction=None):
        event = self.narrator.generate_subsequent_plan(
            self.scene_description,
            self.tags,
            user_instruction,
        )
        if event is None:
            return None, None

        if not self.narrator.event_based_planning:
            parsed_event = event  # already parsed
            dummy_event = self.narrator.previous_events[-1]
            return dummy_event, parsed_event

        parsed_event = self.event_parser.parse_event(
            event,
            self.scene_description,
            self.tags,
            self.narrator.previous_events[:-1],  # since the last event is the current event
        )
        return event, parsed_event

    def allocate_new_plan(self, event: Event, parsed_event):
        # assign the event to the involved characters
        for character in event.involved_characters:
            if character.state != State.IDLE:  # when looping
                self.state_manager.change_state(character, State.IDLE)
                if character.interactee is not None:
                    self.state_manager.change_state(character.interactee, State.IDLE)
            character.event = event

        # update character's information based on the parsed event
        for character in event.involved_characters:
            character.target_actions = parsed_event[character.name]["target_action"]
            character.previous_relationships = character.relationships
            character.relationships = parsed_event[character.name]["relationships"]
            if not character.relationships:
                character.relationships = character.previous_relationships.copy()
        if len(event.involved_characters) == 2:
            character_1 = event.involved_characters[0]
            character_2 = event.involved_characters[1]
            if self._is_planned_to_interact(character_1, character_2):
                character_1.interactee = character_2
                character_2.interactee = character_1

        # infer target positions and target root positions
        position_to_nearby = None
        positions_to_avoid = [c.target_position for c in self.characters if c not in event.involved_characters]
        for character in event.involved_characters:
            character.target_position = self._infer_target_position(
                character,
                position_to_nearby,
                positions_to_avoid,
            )
            position_to_nearby = character.target_position
            positions_to_avoid.append(character.target_position)

            if "sit" in character.target_actions or "lie" in character.target_actions:
                character.target_root_position = self._calculate_target_root_position(character.target_position)
            else:
                character.target_position = self.scene.grid_map.get_closest_grid_position(
                    character.target_position,
                    exclude_occupied=False,
                    return_tensor=True,
                )

        # infer target facing directions
        for character in event.involved_characters:
            character.target_facing_direction = self._infer_target_direction(character)

    def _is_planned_to_interact(self, character_1: Character, character_2: Character):
        try:
            hhi_action = next(action for action in character_1.target_actions if action in self.hhi_tags)
        except StopIteration:
            return False
        return hhi_action in character_2.target_actions

    def _infer_target_position(
        self,
        character: Character,
        position_to_nearby,
        positions_to_avoid,
    ):
        """Before this funciton is called, characters' relationships should be updated beforehand"""

        if self._is_positional_relationships_unchanged(character.previous_relationships, character.relationships):
            return character.position  # keep the current position

        target_position = self.position_sampler.sample_position(
            conditions=character.relationships,
            position_to_nearby=position_to_nearby,
            positions_to_avoid=positions_to_avoid,
        )
        # if sampled position is too close to the character,
        # we prefer not to move the character
        distance = torch.norm(target_position - character.position)
        if distance < self.minimum_distance:
            target_position = character.position
        return target_position

    def _is_positional_relationships_unchanged(self, prev_relationships, new_relationships):
        # check whether the positional relationships are same as the previous ones
        if not prev_relationships:
            return False

        prev_positional_relationships = set(
            relationship
            for relationship in prev_relationships
            if self.position_relationships_pattern.match(relationship)
        )
        new_positional_relationships = set(
            relationship
            for relationship in new_relationships
            if self.position_relationships_pattern.match(relationship)
        )
        return prev_positional_relationships == new_positional_relationships

    def _infer_target_direction(self, character: Character):
        """Before this function is called, all chracaters' target positions should be updated beforehand"""

        target_direction = None
        # if 'look at' relationship is given, we set the target direction towards the anchor
        for relationship in character.relationships:
            if not relationship.startswith("look at"):
                continue

            anchor = relationship.split(" ")[-1]
            anchor = self.finder[anchor]
            if isinstance(anchor, Character):
                anchor_position = anchor.target_position
            else:
                anchor_position = torch.tensor(anchor.center, device=self.device).reshape(1, 3)
            anchor_position[:, 2] = 0.0
            target_direction = anchor_position - character.target_position
            target_direction /= torch.norm(target_direction, dim=-1, keepdim=True)
            break

        # in case of human-scene interaction, we set the anchor's orientation as the target direction
        # or adjust the target direction not to be too far from the anchor's orientation
        for relationship in character.relationships:
            if not (relationship.startswith("sit") or relationship.startswith("lie")):
                continue

            anchor = relationship.split(" ")[-1]
            anchor_orientation = torch.tensor(
                self.finder[anchor].orientation,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)

            if target_direction is None:
                target_direction = anchor_orientation
            else:
                target_direction = misc.calculate_rotated_direction(
                    initial=anchor_orientation,
                    target=target_direction,
                    threshold_angle=torch.pi / 8,
                )
            break

        if target_direction is None:
            target_direction = character.target_facing_direction
        return target_direction

    def _calculate_target_root_position(self, target_position):
        ray_origin = target_position.clone().squeeze().cpu().numpy()
        ray_origin[2] = 2.0
        ray_direction = np.array([[0.0, 0.0, -1.0]])

        locations, _, _ = self.scene.scene_mesh.ray.intersects_location(
            ray_origin[None],
            ray_direction,
        )

        scale = 0.01
        while locations.size == 0:
            new_ray_origin = ray_origin + np.random.rand(3) * scale
            new_ray_origin[2] = 2.0
            locations, _, _ = self.scene.scene_mesh.ray.intersects_location(
                new_ray_origin[None],
                ray_direction,
            )
            scale += 0.01
        locations = locations[locations[:, 2].argmax()].reshape(1, 3)

        target_object: Object = None
        for obj in self.scene.objects:
            if obj.mesh.convex_hull.contains(locations):
                target_object = obj
                break
        self.logger.debug(f"Target Object for Sitting: {target_object}")

        height_offset = target_object.sit_offset_vertical if target_object is not None else self.root_height_offset
        target_root_position = locations + np.array([0, 0, height_offset])
        target_root_position = torch.tensor(target_root_position, device=self.device).reshape(1, 3)
        return target_root_position
