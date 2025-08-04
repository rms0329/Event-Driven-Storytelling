import pickle
from pathlib import Path

import numpy as np
import torch

from . import smpl
from .transform import axis_angle_to_matrix, matrix_to_axis_angle

"""
Inter-X_Dataset
├── LICENSE.md
├── annots
│   ├── action_setting.txt # 40 action categories
│   ├── big_five.npy # big-five personalities
│   ├── familiarity.txt # familiarity level, from 1-4, larger means more familiar
│   └── interaction_order.pkl # actor-reactor order, 0 means P1 is actor; 1 means P2 is actor
├── splits # train/val/test splittings
│   ├── all.txt
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── motions.zip # SMPL-X parameters at 120 fps
├── skeletons.zip # skeleton parameters at 120 fps
└── texts.zip # textual descriptions

### Number of motion clips by gender (actor, reactor)
- ('female', 'female'): 3798
- ('male', 'male'): 3623
- ('female', 'male'): 2099
- ('male', 'female'): 1868
"""

PROXIMITY_REQUIRED_ACTIONS = [
    "hug",
    "handshake",
    "grab",
    "hit",
    "kick",
    "push",
    "pull",
    "sit on leg",
    "slap",
    "pat on back",
    "walk towards",
    "knock over",
    "step on foot",
    "high-five",
    "chase",
    "whisper in ear",
    "support with hand",
    "dance",
    "link arms",
    "shoulder to shoulder",
    "carry on back",
    "massaging shoulder",
    "massaging leg",
    "hand wrestling",
    "pat on cheek",
    "touch head",
    "kiss on cheek",
    "help up",
    "cover mouth",
    "block",
]

_BASE_DIR = Path("./data/Inter-X")
_ANNOTATION_DIR = _BASE_DIR / "annots"
_MOTION_DIR = _BASE_DIR / "motions"
_TEXT_DIR = _BASE_DIR / "texts"
_SPLIT_DIR = _BASE_DIR / "splits"
_PROBLEMATIC_MOTION_IDS = [
    "G055T003A016R008",  # too many frames to use (13884 frames)
]


def get_motion_ids(split="all"):
    with open(_SPLIT_DIR / f"{split}.txt", "r") as f:
        ids = f.readlines()

    ids = [id_.strip() for id_ in ids]
    for problematic_motion_id in _PROBLEMATIC_MOTION_IDS:
        ids.remove(problematic_motion_id)
    return ids


def get_motions(motion_id):
    p1_data = np.load(_MOTION_DIR / motion_id / "P1.npz")
    p2_data = np.load(_MOTION_DIR / motion_id / "P2.npz")
    with open(_ANNOTATION_DIR / "interaction_order.pkl", "rb") as f:
        interaction_order = pickle.load(f)

    # TODO: need to check
    # unlike the official explanation, it seems the interaction order is reversed..
    # if the order is 0, P2 is the actor, not the P1
    if interaction_order[motion_id] == 0:
        return (p2_data, p1_data)
    elif interaction_order[motion_id] == 1:
        return (p1_data, p2_data)
    else:
        raise ValueError("Invalid interaction order found")


def get_motion_infos(motion_data):
    assert motion_data[0]["trans"].shape[0] == motion_data[1]["trans"].shape[0]
    num_frames = motion_data[0]["trans"].shape[0]
    framerate = 120
    genders = (
        str(motion_data[0]["gender"]),
        str(motion_data[1]["gender"]),
    )
    return num_frames, framerate, genders


def construct_body_params(
    motion_data,
    device="cuda",
    use_zero_betas=True,
    genders=None,
):
    actor_data = motion_data[0]
    reactor_data = motion_data[1]
    num_frames, framerate, original_genders = get_motion_infos(motion_data)
    if genders is None:
        genders = original_genders

    bp_actor = {
        "transl": actor_data["trans"],
        "global_orient": actor_data["root_orient"],
        "body_pose": actor_data["pose_body"].reshape(-1, 63),
        "left_hand_pose": actor_data["pose_lhand"].reshape(-1, 45),
        "right_hand_pose": actor_data["pose_rhand"].reshape(-1, 45),
        "betas": np.repeat(actor_data["betas"], repeats=num_frames, axis=0),
    }
    bp_actor = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in bp_actor.items()}
    bp_reactor = {
        "transl": reactor_data["trans"],
        "global_orient": reactor_data["root_orient"],
        "body_pose": reactor_data["pose_body"].reshape(-1, 63),
        "left_hand_pose": reactor_data["pose_lhand"].reshape(-1, 45),
        "right_hand_pose": reactor_data["pose_rhand"].reshape(-1, 45),
        "betas": np.repeat(reactor_data["betas"], repeats=num_frames, axis=0),
    }
    bp_reactor = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in bp_reactor.items()}

    if use_zero_betas:
        bp_actor["betas"][...] = 0.0
        bp_reactor["betas"][...] = 0.0

    bp_actor = _transform_to_z_up(bp_actor, genders[0], (use_zero_betas or genders[0] != original_genders[0]))
    bp_reactor = _transform_to_z_up(bp_reactor, genders[1], (use_zero_betas or genders[1] != original_genders[1]))
    return [bp_actor, bp_reactor]


def _transform_to_z_up(body_params, gender, adjust_height: bool):
    """
    Transform the motion so that the upward direction goes from the +y axis to the +z axis.
    """
    device = body_params["transl"].device
    bm = smpl.load_body_model(device=device, gender=gender)
    rotation = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )  # 90 degree around the x-axis
    primal_root_position = bm(**{"betas": body_params["betas"][:1]}).joints[0, 0]  # before applying anything
    original_first_frame_mesh = bm(**{k: v[:1] for k, v in body_params.items()}).vertices[0]
    original_root_trajectory = body_params["transl"] + primal_root_position
    original_root_heights = original_root_trajectory[:, 1].clone()

    # adjust the height to prevent the character from penetrating or levitating against the ground
    # this is required because we may have changed 'beta' or 'gender' for convenience
    if adjust_height:
        original_root_heights -= original_first_frame_mesh[:, 1].min()

    # calculate new transl
    original_root_trajectory[:, 1] = 0.0  # project on ground (xz-plane)
    new_root_trajectory = (rotation @ original_root_trajectory.view(-1, 3, 1)).squeeze(-1)
    new_root_trajectory[:, 2] = original_root_heights
    new_transl = new_root_trajectory - primal_root_position

    # calculate new global_orient
    new_global_orient = matrix_to_axis_angle(
        rotation @ axis_angle_to_matrix(body_params["global_orient"]),
    )

    # replace the transformed body_params
    body_params["transl"] = new_transl
    body_params["global_orient"] = new_global_orient
    return body_params


def get_annotations(motion_id):
    with open(_TEXT_DIR / f"{motion_id}.txt", "r") as f:
        annotations = f.readlines()
        annotations = [anno.strip() for anno in annotations]
    return annotations


def get_action_labels():
    with open(_ANNOTATION_DIR / "action_setting.txt", "r") as f:
        action_labels = f.readlines()
    return [label.strip().lower() for label in action_labels]


def get_action_label(motion_id):
    action_labels = get_action_labels()
    action_id = int(motion_id[9:12])  # action label uses 0-based indexing
    return action_labels[action_id]


def get_action_label_to_id_dict():
    action_labels = get_action_labels()
    return {action_label: action_id for action_id, action_label in enumerate(action_labels)}


def get_personalities(motion_id):
    group_number = int(motion_id[1:4]) - 1  # change from 1-based indexing to 0-based indexing
    personalities = np.load(_ANNOTATION_DIR / "big_five.npy")
    with open(_ANNOTATION_DIR / "interaction_order.pkl", "rb") as f:
        interaction_order = pickle.load(f)

    if interaction_order[motion_id] == 0:
        return [personalities[2 * group_number], personalities[2 * group_number + 1]]
    elif interaction_order[motion_id] == 1:
        return [personalities[2 * group_number] + 1, personalities[2 * group_number]]
    else:
        raise ValueError("Invalid interaction order found")


def get_familiarity(motion_id):
    group_number = int(motion_id[1:4]) - 1  # change from 1-based indexing to 0-based indexing
    with open(_ANNOTATION_DIR / "familiarity.txt", "r") as f:
        familiarities = f.readlines()
    return int(familiarities[group_number])


if __name__ == "__main__":
    ids = get_motion_ids()
    print(len(ids))
    print(ids[0:3])

    motion_data = get_motions("G001T000A000R000")
    num_frames, framerate, genders = get_motion_infos(motion_data)
    body_params = construct_body_params(motion_data, device="cpu")
    for k, v in body_params[0].items():
        print(f"{k}: {v.shape}")

    print(get_personalities("G001T000A000R000"))
    print(get_familiarity("G001T000A000R000"))
    print(get_annotations("G001T000A000R000"))
    print(get_action_label("G001T000A000R000"))
