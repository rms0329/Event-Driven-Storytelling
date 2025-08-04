import json
import random

import numpy as np
import smplx
import torch
from scipy.signal import savgol_filter
from smplx.utils import SMPLXOutput

from .transform import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)

_BODY_MODEL = {
    "male": None,
    "female": None,
    "neutral": None,
}

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

CONNECTIVITY = [
    [-1, 0],  # pelvis
    [0, 1],  # 0: pelvis -> left_hip
    [0, 2],  # 1: pelvis -> right_hip
    [0, 3],  # 2: pelvis -> spine1
    [1, 4],  # 3: left_hip -> left_knee
    [2, 5],  # 4: right_hip -> right_knee
    [3, 6],  # 5: spine1 -> spine2
    [4, 7],  # 6: left_knee -> left_ankle
    [5, 8],  # 7: right_knee -> right_ankle
    [6, 9],  # 8: spine2 -> spine3
    [7, 10],  # 9: left_ankle -> left_foot
    [8, 11],  # 10: right_ankle -> right_foot
    [9, 12],  # 11: spine3 -> neck
    [9, 13],  # 12: spine3 -> left_collar
    [9, 14],  # 13: spine3 -> right_collar
    [12, 15],  # 14: neck -> head
    [13, 16],  # 15: left_collar -> left_shoulder
    [14, 17],  # 16: right_collar -> right_shoulder
    [16, 18],  # 17: left_shoulder -> left_elbow
    [17, 19],  # 18: right_shoulder -> right_elbow
    [18, 20],  # 19: left_elbow -> left_wrist
    [19, 21],  # 20: right_elbow -> right_wrist
]

# initial root position when betas are all 0
INITIAL_ROOT_POSITION_MALE = np.array([0.00116772, -0.36684087, 0.01266907])
INITIAL_ROOT_POSITION_FEMALE = np.array([0.00055597, -0.33824965, 0.01352531])


class BodyModelWrapper:
    def __init__(self, num_betas=10, gender="male", device="cuda") -> None:
        self.bm = smplx.create(
            "./data/SMPL/",
            model_type="smplx",
            num_betas=num_betas,
            gender=gender,
            use_pca=False,
        ).to(device)
        self.device = device
        self.required_keys = [
            "transl",
            "global_orient",
            "body_pose",
            "left_hand_pose",
            "right_hand_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "expression",
            "betas",
        ]
        self.body_param_dims = {
            "transl": 3,
            "global_orient": 3,
            "body_pose": 63,
            "left_hand_pose": 45,
            "right_hand_pose": 45,
            "jaw_pose": 3,
            "leye_pose": 3,
            "reye_pose": 3,
            "expression": 10,
            "betas": num_betas,
        }

    def __call__(self, body_params=None, return_numpy=False, **body_params_kwargs) -> SMPLXOutput:
        if body_params is None:
            body_params = body_params_kwargs
        elif isinstance(body_params, torch.Tensor):
            body_params = {
                "transl": body_params[:, :3],
                "global_orient": quaternion_to_axis_angle(body_params[:, 3:7]),
                "body_pose": quaternion_to_axis_angle(body_params[:, 7:].reshape(-1, 21, 4)).reshape(-1, 63),
            }

        key = next(iter(body_params))  # any existing key in the body_params
        num_frames = body_params[key].shape[0]
        for key in self.required_keys:
            if key not in body_params:
                body_params[key] = torch.zeros(
                    (num_frames, self.body_param_dims[key]),
                    dtype=torch.float32,
                    device=self.device,
                )

        output: SMPLXOutput = self.bm(**body_params)
        if return_numpy:
            output.vertices = output.vertices.detach().cpu().numpy()
            output.joints = output.joints.detach().cpu().numpy()
        return output

    def to(self, device):
        self.bm = self.bm.to(device)
        self.device = device
        return self

    @property
    def faces(self):
        return self.bm.faces


def load_body_model(num_betas=10, gender="male", device="cuda") -> BodyModelWrapper:
    global _BODY_MODEL
    if _BODY_MODEL[gender] is None:
        _BODY_MODEL[gender] = BodyModelWrapper(num_betas, gender, device)
    return _BODY_MODEL[gender]


def load_body_segmentation():
    with open("data/SMPL/smplx_vert_segmentation.json", "r") as f:
        return json.load(f)


def get_joint_idx(joint_name):
    return JOINT_NAMES.index(joint_name)


def get_joint_indices(joint_names, device="cuda"):
    return torch.tensor([get_joint_idx(j) for j in joint_names], dtype=torch.int32, device=device)


def concatenate_body_params(*body_params):
    combined_body_params = {
        "transl": torch.cat([bp["transl"] for bp in body_params], dim=0),
        "global_orient": torch.cat([bp["global_orient"] for bp in body_params], dim=0),
        "body_pose": torch.cat([bp["body_pose"] for bp in body_params], dim=0),
        "left_hand_pose": torch.cat([bp["left_hand_pose"] for bp in body_params], dim=0),
        "right_hand_pose": torch.cat([bp["right_hand_pose"] for bp in body_params], dim=0),
        "betas": torch.cat([bp["betas"] for bp in body_params], dim=0),
    }
    return combined_body_params


def get_device(body_params):
    if isinstance(body_params, dict):
        return body_params["transl"].device
    elif isinstance(body_params, torch.Tensor):
        return body_params.device
    else:
        raise ValueError("Invalid body_params type")


def get_num_frames(body_params):
    if isinstance(body_params, dict):
        return body_params["transl"].shape[0]
    elif isinstance(body_params, torch.Tensor):
        return body_params.shape[0]
    else:
        raise ValueError("Invalid body_params type")


def get_eye_direction(vertices):
    """
    This function expects the vertices from the SMPL-X model
    (vertices from the SMPL-H is not compatible with this function)
    """
    midpoint_eyes = vertices[:, 9004]
    back_of_head = vertices[:, 8980]
    eye_direction = midpoint_eyes - back_of_head
    eye_direction /= torch.norm(eye_direction, dim=1, keepdim=True)
    return eye_direction, midpoint_eyes


def get_local_frames(body_params):
    sagittal_axis = get_facing_directions(body_params)
    num_frames = sagittal_axis.shape[0]
    device = sagittal_axis.device

    # get local frames
    vertical_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(num_frames, 1)
    horizontal_axis = torch.cross(vertical_axis, sagittal_axis, dim=-1)
    horizontal_axis /= torch.norm(horizontal_axis, dim=-1, keepdim=True)

    local_frames = torch.stack([horizontal_axis, sagittal_axis, vertical_axis], dim=1)
    return local_frames  # (num_frames, 3, 3)


def get_facing_directions(body_params, apply_filter=False, filter_window_size=15, filter_poly_order=3):
    if isinstance(body_params, dict):
        global_orients = axis_angle_to_matrix(body_params["global_orient"])
    else:
        global_orients = quaternion_to_matrix(body_params[:, 3:7])
    num_frames = global_orients.shape[0]
    device = global_orients.device

    # get facing direction
    facing_directions = global_orients[:, :, 2]
    facing_directions[..., 2] = 0
    facing_directions /= torch.norm(facing_directions, dim=-1, keepdim=True)

    # apply Savitzky-Golay filter to smooth the facing direction
    if apply_filter and filter_window_size < num_frames:
        facing_directions = savgol_filter(
            facing_directions.cpu().numpy(),
            filter_window_size,
            filter_poly_order,
            axis=0,
        )
        facing_directions = torch.tensor(facing_directions, dtype=torch.float32, device=device)
        facing_directions /= torch.norm(facing_directions, dim=-1, keepdim=True)
    return facing_directions


def get_character_positions(body_params, apply_filter=False, filter_window_size=15, filter_poly_order=3):
    if isinstance(body_params, dict):
        transl = body_params["transl"]
    else:
        transl = body_params[:, :3]
    num_frames = transl.shape[0]
    device = transl.device

    init_root_position = torch.tensor(INITIAL_ROOT_POSITION_MALE, dtype=torch.float32, device=device)
    character_positions = transl + init_root_position
    character_positions[..., 2] = 0  # project to the ground

    # apply Savitzky-Golay filter to smooth the character positions
    if apply_filter and filter_window_size < num_frames:
        character_positions = savgol_filter(
            character_positions.cpu().numpy(),
            filter_window_size,
            filter_poly_order,
            axis=0,
        )
        character_positions = torch.tensor(character_positions, dtype=torch.float32, device=device)
    return character_positions


def get_init_root_position(body_params, gender):
    if isinstance(body_params, dict):
        beta = body_params["betas"][:1]
    else:
        # beta = body_params[:1, -10:]
        raise NotImplementedError()

    bm = load_body_model(gender=gender, device=beta.device)
    init_root_position = bm(**{"betas": beta}).joints[0, 0]  # before applying anything
    return init_root_position


def get_root_positions(body_params):
    if isinstance(body_params, dict):
        transl = body_params["transl"]
    else:
        transl = body_params[:, :3]

    init_root_position = torch.tensor(INITIAL_ROOT_POSITION_MALE, dtype=torch.float32, device=transl.device)
    root_positions = transl + init_root_position
    return root_positions


def get_character_rotations(facing_directions):
    num_frames = facing_directions.shape[0]
    device = facing_directions.device
    angles = torch.atan2(facing_directions[:, 1], facing_directions[:, 0])
    rotations = axis_angle_to_quaternion(
        torch.stack(
            [
                torch.zeros(num_frames, device=device),
                torch.zeros(num_frames, device=device),
                angles,
            ],
            dim=-1,
        )
    )
    return rotations


# code from 'https://github.com/naver/posescript'
def canonicalize_body_params(body_params):
    def rotvec_to_eulerangles(x):
        x_rotmat = axis_angle_to_matrix(x)
        thetax = torch.atan2(x_rotmat[:, 2, 1], x_rotmat[:, 2, 2])
        thetay = torch.atan2(
            -x_rotmat[:, 2, 0],
            torch.sqrt(x_rotmat[:, 2, 1] ** 2 + x_rotmat[:, 2, 2] ** 2),
        )
        thetaz = torch.atan2(x_rotmat[:, 1, 0], x_rotmat[:, 0, 0])
        return thetax, thetay, thetaz

    def eulerangles_to_rotvec(thetax, thetay, thetaz):
        N = thetax.numel()
        # rotx
        rotx = torch.eye((3)).to(thetax.device).repeat(N, 1, 1)
        roty = torch.eye((3)).to(thetax.device).repeat(N, 1, 1)
        rotz = torch.eye((3)).to(thetax.device).repeat(N, 1, 1)
        rotx[:, 1, 1] = torch.cos(thetax)
        rotx[:, 2, 2] = torch.cos(thetax)
        rotx[:, 1, 2] = -torch.sin(thetax)
        rotx[:, 2, 1] = torch.sin(thetax)
        roty[:, 0, 0] = torch.cos(thetay)
        roty[:, 2, 2] = torch.cos(thetay)
        roty[:, 0, 2] = torch.sin(thetay)
        roty[:, 2, 0] = -torch.sin(thetay)
        rotz[:, 0, 0] = torch.cos(thetaz)
        rotz[:, 1, 1] = torch.cos(thetaz)
        rotz[:, 0, 1] = -torch.sin(thetaz)
        rotz[:, 1, 0] = torch.sin(thetaz)
        rotmat = torch.einsum("bij,bjk->bik", rotz, torch.einsum("bij,bjk->bik", roty, rotx))
        return matrix_to_axis_angle(rotmat)

    initial_rotations = body_params["global_orient"]
    thetax, thetay, thetaz = rotvec_to_eulerangles(initial_rotations)
    zeros = torch.zeros_like(thetaz)
    canonicalized_rotations = eulerangles_to_rotvec(thetax, thetay, zeros)
    return {
        "transl": torch.zeros_like(body_params["transl"]),
        "global_orient": canonicalized_rotations,
        "body_pose": body_params["body_pose"],
        "betas": torch.zeros_like(body_params["betas"]),
    }


def transform_to_z_up(body_params, gender, adjust_height: bool):
    """
    Transform the motion so that the upward direction goes from the +y axis to the +z axis.
    """
    device = body_params["transl"].device
    bm = load_body_model(device=device, gender=gender)
    rotation = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )  # 90 degree around the x-axis

    # before applying anything
    betas = body_params["betas"][:1] if "betas" in body_params else torch.zeros(1, 10, device=device)
    primal_root_position = bm(**{"betas": betas}).joints[0, 0]
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
