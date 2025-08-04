from typing import List

import torch

from ..scene.scene import Scene
from ..type import Character
from ..utils import amass, misc, smpl
from ..utils.transform import quaternion_multiply, quaternion_relative_rotation
from .motion_matching.motion_matcher import MotionMatcher
from .path_planning.path_planner import PathPlanner
from .state_manager import StateManager
from .states import State
from .utils import feature as mm


class MotionSynthesisModule:
    def __init__(self, scene: Scene, characters: List[Character], cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.scene = scene
        self.characters = characters

        self.bm = smpl.load_body_model(device=cfg.device)
        self.motion_matcher = MotionMatcher(cfg)
        self.path_planner = PathPlanner(scene.grid_map, cfg)
        self.state_manager = StateManager(self.motion_matcher, scene.grid_map, cfg)

    def get_current_mesh(self, character: Character):
        vertices = self.bm(character.body_params[-1:]).vertices[0].cpu().numpy()
        faces = self.bm.faces
        return vertices, faces

    def get_mesh_at(self, character: Character, fId: int):
        if fId < 0 or fId >= len(character.body_params):
            raise ValueError(f"Frame index {fId} is out of bounds for character {character.name}.")
        vertices = self.bm(character.body_params[fId : fId + 1]).vertices[0].cpu().numpy()
        faces = self.bm.faces
        return vertices, faces

    def advance_motion_frame(self, character: Character):
        self.state_manager.update_state(character)
        if self.motion_matcher.is_motion_update_needed(character):
            if character.state in [State.IDLE, State.APPROACHING]:
                self.path_planner.update_character_path(character, self.characters)
                self.path_planner.update_future_prediction(character)
            self.motion_matcher.update_best_matching_frame(character)
        self.motion_matcher.advance_motion_frame(character)

    def create_initial_body_params(
        self,
        position,
        facing_direction,
        motion_file="ACCAD/Male1General_c3d/General A1 - Stand_poses.npz",
    ):
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, device=self.device, dtype=torch.float32)
        if not isinstance(facing_direction, torch.Tensor):
            facing_direction = torch.tensor(facing_direction, device=self.device, dtype=torch.float32)
            if facing_direction.shape[-1] == 2:
                facing_direction = misc.add_z(facing_direction)

        motion_data = amass.get_motion(motion_file)
        body_params = amass.construct_body_params(motion_data, device=self.device)
        body_params = mm.to_tensor(body_params)
        body_params = body_params[:3]  # take only the first 3 frames for initialization

        init_directions = smpl.get_facing_directions(body_params)
        init_rotations = smpl.get_character_rotations(init_directions)
        target_rotation = smpl.get_character_rotations(facing_direction.reshape(1, 3))

        base_root_position = torch.tensor(smpl.INITIAL_ROOT_POSITION_MALE, device=self.device, dtype=torch.float32)
        body_params[:, :2] = position[:2] - base_root_position[:2]
        body_params[:, 3:7] = quaternion_multiply(
            quaternion_relative_rotation(init_rotations, target_rotation),
            body_params[:, 3:7],
        )
        return body_params
