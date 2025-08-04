from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from src.type import Character, Event, Role
from src.utils import misc

from .utils import feature as mm

if TYPE_CHECKING:
    from .state_manager import StateManager


class State(IntEnum):
    IDLE = 0
    APPROACHING = 1
    INTERACTING = 2
    TRANSITION_IN = 3
    TRANSITION_OUT = 4


class StateBase:
    def __init__(self, manager: StateManager) -> None:
        self.manager = manager
        self.motion_matcher = manager.motion_matcher
        self.grid_map = manager.grid_map
        self.motion_db = self.motion_matcher.motion_db
        self.hhi_tags = self.motion_db.hhi_tags
        self.hsi_tags = self.motion_db.hsi_tags
        self.hsi_transition_done_delay = manager.hsi_transition_done_delay
        self.logger = manager.logger

    def enter(self, character: Character, prev_state):
        pass

    def update(self, character: Character):
        raise NotImplementedError

    def exit(self, character: Character, next_state):
        pass

    def __eq__(self, value) -> bool:
        if isinstance(value, str):
            class_name = self.__class__.__name__.lower()
            return class_name.startswith(value.lower().replace("_", ""))
        if isinstance(value, State):  # compare with State enum via string
            class_name = self.__class__.__name__
            enum_name = value.name.title().replace("_", "")
            return class_name.startswith(enum_name)
        if isinstance(value, StateBase):
            return self.__class__ == value.__class__
        raise NotImplementedError

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return class_name.replace("State", "").lower()

    @property
    def fId(self):
        return self.manager.fId

    def try_match_from_human_interaction_db(self, character: Character):
        # match cannot be done when either character or interactee is in transition states
        if character.state in [State.TRANSITION_IN, State.TRANSITION_OUT]:
            return False
        if character.interactee.state in [State.TRANSITION_IN, State.TRANSITION_OUT]:
            return False

        # match cannot be done when either character or interactee
        # has not yet made a complete human-scene interaction
        if character.is_planned_to_interact_with_scene() and character.state == State.APPROACHING:
            return False
        if character.interactee.is_planned_to_interact_with_scene() and character.interactee.state == State.APPROACHING:
            return False

        if character.is_resting_posture() and character.interactee.is_resting_posture():
            return character.state == State.IDLE and character.interactee.state == State.IDLE

        if character.state == State.APPROACHING and character.interactee.state == State.APPROACHING:
            return False

        distance_to_interactee = torch.norm(character.position - character.interactee.position)  # fmt: skip
        if distance_to_interactee > self.manager.hhi_matching_start_distance:
            return False

        tag = next(a for a in character.target_actions if a in self.hhi_tags)
        fId = self.motion_db.search_best_matching_frame(character, tag)
        matched_position = self.motion_db.get_matching_feature(fId, tag, "relative_position")  # fmt: skip
        matched_position = mm.to_world(
            misc.add_z(matched_position, z=0),
            character.position,
            character.rotation,
        )
        matched_direction = self.motion_db.get_matching_feature(fId, tag, "relative_direction")  # fmt: skip
        matched_direction = mm.to_world(
            misc.add_z(matched_direction, z=0),
            character.rotation,
        )

        matching_distance = torch.norm(character.interactee.position - matched_position) + torch.norm(
            character.interactee.facing_direction - matched_direction
        )
        if matching_distance <= self.manager.hhi_matching_threshold:
            return True
        return False

    def try_match_from_scene_interaction_db(self, character: Character):
        # to prevent the cases where a character matches from the back of the chair/sofa
        # maybe considering the relative direction between the target direction and character position would be better..?
        if len(character.grid_path) > self.manager.hsi_grid_path_len_threshold:
            return False

        tag = next(a for a in character.target_actions if a in self.hsi_tags)
        tag += "_transition_in"
        fId = self.motion_db.search_best_matching_frame(character, tag)
        matched_position = self.motion_db.get_matching_feature(fId, tag, "relative_position")  # fmt: skip
        matched_direction = self.motion_db.get_matching_feature(fId, tag, "relative_direction")  # fmt: skip
        matched_height = self.motion_db.get_matching_feature(fId, tag, "root_height")  # fmt: skip

        matched_root_position = torch.cat([matched_position, matched_height], dim=1)
        matched_root_position = mm.to_world(
            matched_root_position,
            character.position,
            character.rotation,
        )
        matched_direction = mm.to_world(
            misc.add_z(matched_direction, z=0),
            character.rotation,
        )
        matching_distance = torch.norm(character.target_root_position - matched_root_position) + torch.norm(
            character.target_facing_direction - matched_direction
        )
        if matching_distance <= self.manager.hsi_matching_threshold:
            character.matched_root_position = matched_root_position
            self.logger.debug(f"HSI matching succeeded ({matching_distance.item():.2f})")
            return True
        self.logger.debug(f"HSI matching failed ({matching_distance.item():.2f})")
        return False


class IdleState(StateBase):
    def enter(self, character: Character, prev_state):
        if "sit" in character.current_actions:
            character.current_actions = ["sit"]
        elif "lie" in character.current_actions:
            character.current_actions = ["lie"]
        else:
            character.current_actions = []

        # this helps to reduce the jitterative motion when the character is idle
        if character.target_facing_direction is None:
            character.target_facing_direction = character.facing_direction

    def update(self, character: Character):
        # character remains idle until the next action plan is set
        if not character.target_actions:
            return

        distance_to_target_position = torch.norm(character.position - character.target_position)

        # when the target position is changed to far from the current position,
        # the character prioritizes approaching the target position
        if distance_to_target_position >= self.manager.far_threshold:
            if "sit" in character.current_actions or "lie" in character.current_actions:
                return self.manager.transition_out_state
            return self.manager.approaching_state

        # when a character doesn't need to wait for someone to execute an action
        if character.interactee is None:
            return self.manager.interacting_state

        # if a character is waiting for the interactee and them is close enough,
        # we try to match interaction motion from the human-human interaction dataset
        if (
            character.interactee is not None
            and character.num_played_frames >= self.motion_matcher.motion_update_frequency
        ):
            matched = self.try_match_from_human_interaction_db(character)
            if matched:
                character.role = Role.ACTOR
                character.interactee.role = Role.REACTOR
                self.manager.change_state(
                    character.interactee,
                    self.manager.interacting_state,
                    enforce_motion_research=False,
                )
                return self.manager.interacting_state

    def exit(self, character: Character, next_state):
        # these are hacks for mouse-interacting demo
        if character.target_actions == ["stand"]:
            character.target_actions = []
        if character.event is None:
            character.event = Event([], "dummy")


class ApproachingState(StateBase):
    def enter(self, character: Character, prev_state):
        character.current_actions = ["walk"]

    def update(self, character: Character):
        distance_to_target_position = torch.norm(character.position - character.target_position)

        if character.is_planned_to_interact_with_scene():
            # when the character is intended to sit or lie down on some object,
            # we need to frequently check whether there is a matched animation in the human-scene interaction database
            if (
                distance_to_target_position <= self.manager.hsi_matching_start_distance
                and character.num_played_frames >= self.motion_matcher.motion_update_frequency
            ):
                matched = self.try_match_from_scene_interaction_db(character)
                if matched:
                    return self.manager.transition_in_state

        elif character.is_planned_to_interact_with_human():
            # when the character is intended to interact with another character,
            # we need to frequently check whether there is a matched animation in the human-human interaction database
            if character.num_played_frames >= self.motion_matcher.motion_update_frequency:
                matched = self.try_match_from_human_interaction_db(character)
                if matched:
                    character.role = Role.ACTOR
                    character.interactee.role = Role.REACTOR
                    self.manager.change_state(
                        character.interactee,
                        self.manager.interacting_state,
                        enforce_motion_research=False,
                    )
                    return self.manager.interacting_state

            # when a character reached to the target position without matching,
            # the character waits its interactee in the idle state
            if distance_to_target_position < self.manager.near_threshold:
                return self.manager.idle_state

        else:
            if distance_to_target_position < self.manager.near_threshold:
                if character.target_actions:
                    return self.manager.interacting_state

                character.event.update_completed(character, self.fId)
                return self.manager.idle_state

    def exit(self, character: Character, next_state):
        character.current_actions.remove("walk")


class InteractingState(StateBase):
    def enter(self, character: Character, prev_state):
        character.current_actions += character.target_actions
        character.current_actions = list(set(character.current_actions))  # remove duplicates
        character.target_actions = []

        for i, action in enumerate(character.current_actions):
            if action in self.motion_db.hhi_tags and character.interactee is None:
                character.current_actions[i] = f"{action}_single"

        if character.is_resting_posture() and len(character.current_actions) > 1:
            character._use_partwise_mm = True
            character.current_actions = [
                (f"{action}_partwise" if action in self.motion_db.partwise_tags else action)
                for action in character.current_actions
            ]
        if character.interactee is not None and character.role == Role.REACTOR:
            character.play_synchronized_motion = True

    def update(self, character: Character):
        """
        Note: reactor character doesn't update its state by itself.
            So, actor character must handle the reactor's state too (= interactee's state)
        """
        if character.play_synchronized_motion:
            return
        if character.looping:
            return

        fId = character._fIds[0] if character._use_partwise_mm else character.fId
        tag = character.upper_body_action if character._use_partwise_mm else character.main_action
        if not self.motion_db.is_end_of_motion(fId, tag):
            return

        # if the playing animation is finished,
        # if it was looping motion, the character remains in the interacting state
        if tag in self.motion_db.looping_tags:
            if character.interactee is not None:
                character.interactee.looping = True
                character.interactee.event.update_completed(character.interactee, self.fId, is_looping=True)
            character.looping = True
            character.event.update_completed(character, self.fId, is_looping=True)
            return

        # in normal cases, the character goes to the idle state
        else:
            if character.interactee is not None:
                self.manager.change_state(character.interactee, self.manager.idle_state)
            return self.manager.idle_state

    def exit(self, character: Character, next_state):
        if character.interactee is not None:
            character.interactee = None
            character.role = Role.NONE
            character.play_synchronized_motion = False

        character._use_partwise_mm = False
        character.target_position = self.grid_map.get_closest_grid_position(
            character.position,
            exclude_occupied=True,
            return_tensor=True,
        )
        character.current_actions = [a.replace("_partwise", "") for a in character.current_actions]
        character.looping = False
        if character.event is not None:
            character.event.update_completed(character, self.fId)


class TransitionInState(StateBase):
    def enter(self, character: Character, prev_state):
        if "sit" in character.target_actions:
            character.current_actions += ["sit_transition_in"]
            character.target_actions.remove("sit")
        elif "lie" in character.target_actions:
            character.current_actions += ["lie_transition_in"]
            character.target_actions.remove("lie")

    def update(self, character: Character):
        if self.motion_db.is_end_of_motion(character.fId, character.main_action):
            self.logger.debug(f"{character.name}: transition done at fId {self.fId} due to the end of motion")
            return self._proceed_to_next_state(character)

        root_distance = torch.norm(character.target_root_position - character.root_position)
        if root_distance > self.manager.hsi_matching_done_root_distance:
            return
        if character.transition_done_fId is None:
            character.transition_done_fId = self.fId
            self.logger.debug(f"{character.name}: transition delay started at fId {self.fId}")
            return
        if self.fId - character.transition_done_fId < self.hsi_transition_done_delay:
            return

        self.logger.debug(f"{character.name}: transition done at fId {self.fId}")
        return self._proceed_to_next_state(character)

    def exit(self, character: Character, next_state):
        character.transition_done_fId = None
        character.accumulated_adjustment = None
        if "sit_transition_in" in character.current_actions:
            character.current_actions.remove("sit_transition_in")
            character.current_actions += ["sit"]
        elif "lie_transition_in" in character.current_actions:
            character.current_actions.remove("lie_transition_in")
            character.current_actions += ["lie"]

    def _proceed_to_next_state(self, character: Character):
        if not character.target_actions:
            character.event.update_completed(character, self.fId)
            return self.manager.idle_state

        if character.interactee is None:
            return self.manager.interacting_state

        # state will be eventually changed to the interacting state
        # this is because more detailed conditions are checked in the idle state (e.g. HHI matching)
        self.manager.change_state(character, self.manager.idle_state)
        return self.manager.update_state(character)


class TransitionOutState(StateBase):
    def enter(self, character: Character, prev_state):
        if "sit" in character.current_actions:
            character.current_actions = ["sit_transition_out"]
        elif "lie" in character.current_actions:
            character.current_actions = ["lie_transition_out"]

    def update(self, character: Character):
        height = character.root_position[0, 2]
        if height >= self.manager.height_threshold:
            return self.manager.approaching_state

    def exit(self, character: Character, next_state):
        if "sit_transition_out" in character.current_actions:
            character.current_actions.remove("sit_transition_out")
        if "lie_transition_out" in character.current_actions:
            character.current_actions.remove("lie_transition_out")
