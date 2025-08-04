from typing import Union

from src.scene.grid_map import GridMap
from src.type import Character
from src.utils import misc

from .motion_matching.motion_matcher import MotionMatcher
from .states import (
    ApproachingState,
    IdleState,
    InteractingState,
    State,
    StateBase,
    TransitionInState,
    TransitionOutState,
)


class StateManager:
    def __init__(self, motion_matcher: MotionMatcher, grid_map: GridMap, cfg) -> None:
        self.cfg = cfg
        self.motion_matcher = motion_matcher
        self.grid_map = grid_map
        self.fId = 0
        self.far_threshold = cfg.state_manager.far_threshold
        self.near_threshold = cfg.state_manager.near_threshold
        self.hsi_matching_start_distance = cfg.state_manager.hsi_matching_start_distance
        self.hsi_matching_threshold = cfg.state_manager.hsi_matching_threshold
        self.hsi_matching_done_root_distance = cfg.state_manager.hsi_matching_done_root_distance
        self.hsi_transition_done_delay = cfg.state_manager.hsi_transition_done_delay
        self.hsi_grid_path_len_threshold = cfg.state_manager.hsi_grid_path_len_threshold
        self.height_threshold = cfg.state_manager.height_threshold
        self.hhi_matching_start_distance = cfg.state_manager.hhi_matching_start_distance
        self.hhi_matching_threshold = cfg.state_manager.hhi_matching_threshold
        self.logger = misc.get_console_logger("StateManager", level=cfg.logging_level)

        self.idle_state = IdleState(self)
        self.approaching_state = ApproachingState(self)
        self.interacting_state = InteractingState(self)
        self.transition_in_state = TransitionInState(self)
        self.transition_out_state = TransitionOutState(self)

        # change state enum to state class
        self._state_enum_to_class = {
            State.IDLE: self.idle_state,
            State.APPROACHING: self.approaching_state,
            State.INTERACTING: self.interacting_state,
            State.TRANSITION_IN: self.transition_in_state,
            State.TRANSITION_OUT: self.transition_out_state,
        }

    def update_state(self, character: Character):
        if character.state is None:
            character.state = self.idle_state

        next_state = character.state.update(character)
        current_state = character.state  # update() may change the state
        if next_state is not None and current_state != next_state:
            current_state.exit(character, next_state=next_state)
            character.state = next_state
            character.enforce_motion_search = True
            character.invoke_inertialization = True
            next_state.enter(character, prev_state=current_state)
            self.logger.info(f"{character.name}: {current_state} -> {character.state}")
            self.logger.debug(f"  - current_action: {character.current_actions}")
            self.logger.debug(f"  - target_action: {character.target_actions}")

    def change_state(
        self,
        character: Character,
        new_state: Union[StateBase, State],
        enforce_motion_research=True,
    ):
        if isinstance(new_state, State):
            new_state = self._state_enum_to_class[new_state]

        current_state = character.state
        if current_state != new_state:
            current_state.exit(character, next_state=new_state)
            character.state = new_state
            character.enforce_motion_search = enforce_motion_research
            character.invoke_inertialization = True
            new_state.enter(character, prev_state=current_state)
            self.logger.info(f"{character.name}: {current_state} -> {character.state}")
            self.logger.debug(f"  - current_action: {character.current_actions}")
            self.logger.debug(f"  - target_action: {character.target_actions}")
