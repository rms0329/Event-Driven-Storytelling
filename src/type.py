from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, ClassVar, List

import numpy as np
import torch
import trimesh

from .utils import smpl
from .utils.transform import quaternion_relative_rotation, quaternion_to_axis_angle

if TYPE_CHECKING:
    from src.motion_synthesis.states import StateBase


class Gender(IntEnum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, gender):
        if gender.lower() == "male":
            return cls.MALE
        elif gender.lower() == "female":
            return cls.FEMALE
        else:
            raise ValueError("Invalid gender")


class Role(IntEnum):
    NONE = -1
    ACTOR = 0
    REACTOR = 1


@dataclass
class Event:
    involved_characters: list[Character]
    activity: str
    state: str = "ongoing"
    completed: list[bool] = field(default_factory=list)
    parsed_event = None
    reasoning: str = None
    created_fId: int = None
    completed_fId: int = None

    def __post_init__(self):
        self.activity = self.activity.rstrip(".")
        self.completed = [False] * len(self.involved_characters)

    def __repr__(self) -> str:
        if isinstance(self.involved_characters[0], str):
            return f"[{', '.join(c for c in self.involved_characters)}] {self.activity}"
        return f"[{', '.join(c.name for c in self.involved_characters)}] {self.activity}"

    def __eq__(self, value) -> bool:
        if isinstance(value, str):
            return str(self) == value
        if isinstance(value, Event):
            return str(self) == str(value)
        raise NotImplementedError("Only string and Event comparison is supported.")

    def __hash__(self) -> int:
        return hash(self.content)

    @property
    def is_group_event(self):
        return len(self.involved_characters) > 1

    @property
    def done(self):
        return all(self.completed)

    def update_completed(self, character: Character, fId: int, is_looping=False):
        idx = self.involved_characters.index(character)
        self.completed[idx] = True
        if not all(self.completed):
            return

        if is_looping:
            self.state = "looping"
        else:
            self.state = "completed"
            self.completed_fId = fId
            for c in self.involved_characters:
                c.event = None


@dataclass
class Character:
    name: str
    gender: Gender = Gender.MALE
    state: StateBase = None
    current_actions: list = field(default_factory=list)
    target_actions: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    previous_relationships: list = field(default_factory=list)
    event: Event = None
    interactee: Character = None
    role: Role = Role.NONE
    dt: float = 1 / 30
    looping: bool = False

    # for path planning
    velocity_mps: float = 1.0
    halflife_position: float = 0.1
    halflife_rotation: float = 0.3
    target_position: torch.Tensor = None
    spline_curve: torch.Tensor = None
    grid_path: list = field(default_factory=list)

    # for motion matching
    fId: int = -1
    motion_db: str = "base"
    num_played_frames: int = 9999
    play_synchronized_motion: bool = False
    enforce_motion_search: bool = False
    invoke_inertialization: bool = True
    future_positions: torch.Tensor = None
    future_facing_directions: torch.Tensor = None
    target_facing_direction: torch.Tensor = None
    target_root_position: torch.Tensor = None
    transition_done_fId: int = None
    accumulated_adjustment: torch.Tensor = None
    matched_root_position: torch.Tensor = None
    matching_feature: torch.Tensor = None
    body_params: torch.Tensor = None
    offset_x: torch.Tensor = None
    offset_v: torch.Tensor = None

    # for partwise motion matching
    _use_partwise_mm: bool = False
    _fIds: List[int] = field(default_factory=lambda: [-1, -1])
    _motion_dbs: List[str] = field(default_factory=lambda: ["base", "base"])
    _update_required: List[bool] = field(default_factory=lambda: [False, False])

    _counter: ClassVar[Counter] = Counter()

    def __post_init__(self):
        if self.name[0].islower():
            self._counter[self.name] += 1
            self.name = f"{self.name}_{self._counter[self.name]}"

    def __repr__(self) -> str:
        return f"{self.name} ({self.state})"

    def __eq__(self, __value: object) -> bool:
        if __value is None:
            return False
        if isinstance(__value, str):
            return self.name == __value
        if isinstance(__value, Character):
            return self.name == __value.name
        raise NotImplementedError("Only string and Character comparison is supported.")

    @property
    def position(self):
        """Returns the position of the character. The shape is (1, 3), but the z value is fixed to 0."""
        return smpl.get_character_positions(self.body_params[-1:])

    @property
    def velocity(self):
        """Returns the velocity of the character. The shape is (1, 3), but the z value is fixed to 0."""
        positions = smpl.get_character_positions(self.body_params[-2:])
        return torch.diff(positions, dim=0) / self.dt

    @property
    def acceleration(self):
        """Returns the acceleration of the character. The shape is (1, 3), but the z value is fixed to 0."""
        positions = smpl.get_character_positions(self.body_params[-3:])
        velocities = torch.diff(positions, dim=0) / self.dt
        return torch.diff(velocities, dim=0) / self.dt

    @property
    def rotation(self):
        """Returns the rotation of the character in quaternion. The shape is (1, 4)."""
        facing_direction = smpl.get_facing_directions(self.body_params[-1:])
        return smpl.get_character_rotations(facing_direction)

    @property
    def angular_velocity(self):
        """Returns the angular_velocity of the character. The shape is (1, 3)."""
        facing_directions = smpl.get_facing_directions(self.body_params[-2:])
        rotations = smpl.get_character_rotations(facing_directions)
        return (
            quaternion_to_axis_angle(
                quaternion_relative_rotation(
                    rotations[:1],
                    rotations[1:],
                )
            )
            / self.dt
        )

    @property
    def facing_direction(self):
        """Returns the facing direction of the character. The shape is (1, 3), but the z value is fixed to 0."""
        return smpl.get_facing_directions(self.body_params[-1:])

    @property
    def root_position(self):
        """Returns the root position of the character. The shape is (1, 3)."""
        return smpl.get_root_positions(self.body_params[-1:])

    @property
    def main_action(self):
        if not self.current_actions:
            if self.is_stationary():
                return "idle"
            return "walk"

        if len(self.current_actions) == 1:
            return self.current_actions[0]

        return next(
            action for action in self.current_actions if not action.startswith("sit") and not action.startswith("lie")
        )

    @property
    def upper_body_action(self):
        tag = next(
            action for action in self.current_actions if not action.startswith("sit") and not action.startswith("lie")
        )
        return tag

    @property
    def lower_body_action(self):
        tag = next(action for action in self.current_actions if action.startswith("sit") or action.startswith("lie"))
        return tag

    def is_stationary(self):
        return all(p == self.grid_path[0] for p in self.grid_path)

    def is_resting_posture(self):
        return self.is_sitting() or self.is_lying()

    def is_sitting(self):
        return any(a.startswith("sit") for a in self.current_actions)

    def is_lying(self):
        return any(a.startswith("lie") for a in self.current_actions)

    def is_planned(self):
        return self.target_actions

    def is_planned_to_interact_with_scene(self):
        return "sit" in self.target_actions or "lie" in self.target_actions

    def is_planned_to_interact_with_human(self):
        return self.interactee is not None


@dataclass
class Object:
    id: int
    label: str
    idx: int
    aa_extents: np.ndarray
    orientation: np.ndarray
    has_orientation: bool
    width: float
    depth: float
    height: float
    center: np.ndarray
    bbox: np.ndarray
    mesh: trimesh.Trimesh
    sit_offset_horizontal: float = 0.0
    sit_offset_vertical: float = 0.0

    def __repr__(self) -> str:
        return f"{self.label}_{self.idx}"

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Object):
            return self.id == __value.id
        elif isinstance(__value, str):
            return f"{self.label}_{self.idx}" == __value
        raise NotImplementedError("Only Object comparison is supported.")

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def min_z(self):
        return self.bbox[:, 2].min()

    @property
    def max_z(self):
        return self.bbox[:, 2].max()

    @property
    def corners(self):
        return self.bbox[:4, :2]
