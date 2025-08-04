import pickle
import random
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from src.motion_synthesis.states import State

from . import smpl

_BASE_DIR = Path("data/SAMP/pkl")
_PROBLEMATIC_FILES = [
    "lie_down_5_stageII",  # too short
    "lie_down_6_stageII",  # too short
    "chair_mo_sit2sit_001_stageII",  # performs interaction twice
    "chair_mo_sit2sit001_stageII",  # performs weired transition
    "chair_mo_sit2sit002_stageII",  # performs weired transition
    "table009_stageII",  # puts one leg on the seat
    "table014_stageII",  # puts one leg on the seat
    "sofa013_stageII",  # almost lying
]


class InteractionState(IntEnum):
    APPROACHING = 0
    INTERACTING = 1
    LEAVING = 2

    @classmethod
    def from_character_state(cls, state):
        if state == State.APPROACHING or state == State.TRANSITION_IN:
            return cls.APPROACHING
        elif state == State.TRANSITION_OUT:
            return cls.LEAVING
        else:
            return cls.INTERACTING


class InteractionType(IntEnum):
    SIT = 0
    LIE = 1

    @classmethod
    def from_texts(cls, texts):
        for text in texts:
            if text == "sit" or text == "sitting":
                return cls.SIT
            if text == "lie" or text == "lying":
                return cls.LIE
        raise ValueError(f"There is no valid interaction type from the {texts}")


class InteractionExtractionError(Exception):
    pass


def get_motion_files():
    return sorted(_BASE_DIR.glob("[!.]*.pkl"))


def get_random_motion(with_source=False):
    pkl_files = list(_BASE_DIR.glob("[!.]*.pkl"))
    pkl_file = random.choice(pkl_files)
    with open(pkl_file, "rb") as f:
        motion_data = pickle.load(f, encoding="latin1")
    if with_source:
        print(f"Loading '{pkl_file.name}'")
        return motion_data, pkl_file
    return motion_data


def get_motion(motion_path, with_source=False):
    motion_path = Path(motion_path)
    if _BASE_DIR in motion_path.parents:
        pkl_file = motion_path
    else:
        pkl_file = _BASE_DIR / motion_path

    with open(pkl_file, "rb") as f:
        motion_data = pickle.load(f, encoding="latin1")
    if with_source:
        print(f"Loading '{pkl_file.name}'")
        return motion_data, pkl_file
    return motion_data


def get_motion_info(motion_data):
    num_frames = motion_data["pose_est_trans"].shape[0]
    framerate = float(motion_data["mocap_framerate"])
    interaction_type = (
        InteractionType.LIE if motion_data["mocap_fname"].split("/")[-1].startswith("lie_down") else InteractionType.SIT
    )
    return num_frames, framerate, interaction_type


def construct_body_params(
    motion_data,
    device="cuda",
    return_numpy=False,
    use_zero_betas=True,
    num_betas=10,
    dtype=torch.float32,
):
    num_frames = motion_data["pose_est_trans"].shape[0]
    body_params = {
        "transl": torch.tensor(motion_data["pose_est_trans"], dtype=dtype, device=device),
        "global_orient": torch.tensor(motion_data["pose_est_fullposes"][:, :3], dtype=dtype, device=device),
        "body_pose": torch.tensor(motion_data["pose_est_fullposes"][:, 3:66], dtype=dtype, device=device),
        "betas": torch.tensor(motion_data["shape_est_betas"][:num_betas], dtype=dtype, device=device).tile(
            num_frames, 1
        ),
    }
    if use_zero_betas:
        body_params["betas"][...] = 0.0
    if return_numpy:
        return {k: v.cpu().numpy() for k, v in body_params.items()}
    return body_params


def extract_interaction_information(body_params, eps=0.05, min_samples=200, num_skip_frames=500):
    root_positions = smpl.get_root_positions(body_params)
    facing_directions = smpl.get_facing_directions(body_params)

    labels_per_frame = DBSCAN(eps=eps, min_samples=min_samples).fit(root_positions.cpu().numpy()).labels_
    labels_per_frame[:num_skip_frames] = -1  # Ignore the initial frames
    if np.all(labels_per_frame < 0):
        raise InteractionExtractionError("All frames are classified as noise")

    n_points_per_cluster = np.bincount(labels_per_frame[labels_per_frame >= 0])  # label -1 is noise
    densest_label = np.argmax(n_points_per_cluster)
    interaction_frames = labels_per_frame == densest_label
    interaction_start_frame = np.nonzero(interaction_frames)[0][0]
    interaction_end_frame = np.nonzero(interaction_frames)[0][-1]

    interaction_frames = torch.tensor(interaction_frames, dtype=torch.bool, device=root_positions.device)
    interaction_position = torch.mean(root_positions[interaction_frames], dim=0, keepdim=True)
    interaction_direction = torch.mean(facing_directions[interaction_frames], dim=0, keepdim=True)
    return (
        interaction_position,
        interaction_direction,
        interaction_start_frame,
        interaction_end_frame,
    )
