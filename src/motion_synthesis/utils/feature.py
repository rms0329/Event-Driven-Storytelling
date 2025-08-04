import torch

from src.utils import smpl
from src.utils.transform import (
    axis_angle_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_relative_rotation,
    quaternion_to_axis_angle,
)


def get_future_positions(body_params, future_offsets, character_positions=None, character_rotations=None):
    if character_positions is None:
        character_positions = smpl.get_character_positions(body_params)
    if character_rotations is None:
        facing_directions = smpl.get_facing_directions(body_params)
        character_rotations = smpl.get_character_rotations(facing_directions)

    future_positions, T_ = aggregate_future_features(character_positions, future_offsets)
    future_positions = to_local(future_positions, character_positions[:T_], character_rotations[:T_])
    return future_positions


def get_future_keyjoint_positions(
    body_params,
    future_offsets,
    keyjoint_ids,
    character_positions=None,
    character_rotations=None,
):
    if character_positions is None:
        character_positions = smpl.get_character_positions(body_params)
    if character_rotations is None:
        facing_directions = smpl.get_facing_directions(body_params)
        character_rotations = smpl.get_character_rotations(facing_directions)

    device = _get_device(body_params)
    bm = smpl.load_body_model(device=device)

    joints = bm(**body_params).joints
    root_positions = joints[:, 0]
    keyjoint_positions = joints[:, keyjoint_ids]  # (T, K, 3)
    future_keyjoint_positions, T_ = aggregate_future_features(keyjoint_positions, future_offsets)
    future_keyjoint_positions = to_local(future_keyjoint_positions, root_positions[:T_], character_rotations[:T_])
    return future_keyjoint_positions  # (T', O, K, 3)


def get_future_facing_directions(body_params, future_offsets, facing_directions=None, character_rotations=None):
    if facing_directions is None:
        facing_directions = smpl.get_facing_directions(body_params)
    if character_rotations is None:
        character_rotations = smpl.get_character_rotations(facing_directions)

    future_facing_directions, T_ = aggregate_future_features(facing_directions, future_offsets)
    future_facing_directions = to_local(future_facing_directions, character_rotations[:T_])
    return future_facing_directions


def get_keyjoint_features(
    body_params,
    keyjoint_ids,
    character_positions=None,
    character_rotations=None,
    dt=1 / 30,
):
    if character_positions is None:
        character_positions = smpl.get_character_positions(body_params)
    if character_rotations is None:
        facing_directions = smpl.get_facing_directions(body_params)
        character_rotations = smpl.get_character_rotations(facing_directions)

    device = _get_device(body_params)
    bm = smpl.load_body_model(device=device)

    joints = bm(**body_params).joints
    root_positions = joints[:, 0]
    keyjoint_positions = joints[:, keyjoint_ids]
    keyjoint_velocities = torch.diff(keyjoint_positions, dim=0, prepend=keyjoint_positions[:1]) / dt

    # keyjoints are localized based on the character's root position and facing direction
    keyjoint_positions = to_local(keyjoint_positions, root_positions, character_rotations)
    keyjoint_velocities = to_local(keyjoint_velocities, character_rotations)
    return keyjoint_positions, keyjoint_velocities


def aggregate_future_features(features, frame_offsets):
    """Get future features for each frame based on given frame offsets.

    :param features: Tensor of shape (T, D) representing the features, where T is the number of frames.
    :param frame_offsets: List of frame offsets.
    :return: Tensor of shape (T', num_offsets, D) representing the future positions of features.
            Note T' < T as the frames close to the end of the sequence cannot have future positions.
    """
    num_frames = features.shape[0]
    device = features.device

    T_ = int(num_frames - max(frame_offsets))
    frames_after_offsets = torch.stack(
        [
            torch.arange(
                offset,
                offset + T_,
                dtype=torch.int64,
                device=device,
            )
            for offset in frame_offsets
        ],
        dim=-1,
    )  # (T', num_offsets)
    return features[frames_after_offsets], T_  # (T', num_offsets, D)


def get_pose_features(body_params, dt=1 / 30):
    device = _get_device(body_params)
    bm = smpl.load_body_model(device=device)
    joints = bm(**body_params).joints

    facing_directions = smpl.get_facing_directions(body_params)
    character_positions = smpl.get_character_positions(body_params)
    character_rotations = smpl.get_character_rotations(facing_directions)

    # character's velocities and angular velocities
    # (should be represented in the local coordinate of the "previous frame")
    character_velocities = torch.diff(character_positions, dim=0, prepend=character_positions[:1]) / dt
    character_velocities = to_local(
        character_velocities,
        torch.cat([character_rotations[:1], character_rotations[:-1]], dim=0),
    )
    character_angular_velocities = (
        quaternion_to_axis_angle(
            quaternion_relative_rotation(
                torch.cat([character_rotations[:1], character_rotations[:-1]], dim=0),
                character_rotations,
            )
        )
        / dt
    )

    # character's root positions and velocities in the local coordinate of the "current frame"
    root_positions = joints[:, 0]
    root_velocities = torch.diff(root_positions, dim=0, prepend=root_positions[:1]) / dt
    root_positions = to_local(root_positions, character_positions, character_rotations)
    root_velocities = to_local(root_velocities, character_rotations)

    # character's root rotations and angular velocities in the local coordinate of the "current frame"
    root_rotations = axis_angle_to_quaternion(body_params["global_orient"])
    root_angular_velocities = (
        quaternion_to_axis_angle(
            quaternion_relative_rotation(
                torch.cat([root_rotations[:1], root_rotations[:-1]], dim=0),
                root_rotations,
            )
        )
        / dt
    )
    root_rotations = to_local(root_rotations, character_rotations)
    root_angular_velocities = to_local(root_angular_velocities, character_rotations)

    # character's joint rotations and angular velocities
    joint_rotations = axis_angle_to_quaternion(body_params["body_pose"].reshape(-1, 21, 3))
    joint_angular_velocities = (
        quaternion_to_axis_angle(
            quaternion_relative_rotation(
                torch.cat([joint_rotations[:1], joint_rotations[:-1]], dim=0),
                joint_rotations,
            )
        )
        / dt
    )

    pose_features = torch.cat(
        [
            character_velocities,  # :3
            character_angular_velocities,  # 3:6
            root_positions,  # 6:9
            root_velocities,  # 9:12
            root_rotations,  # 12:16
            root_angular_velocities,  # 16:19
            joint_rotations.reshape(-1, 84),  # 19:103
            joint_angular_velocities.reshape(-1, 63),  # 103:166
        ],
        dim=-1,
    )
    assert pose_features.shape[-1] == 166, f"{pose_features.shape[-1]}"
    return pose_features


def advance_frame(body_params: torch.Tensor, pose_feature: torch.Tensor, dt=1 / 30):
    if pose_feature.ndim == 1:
        pose_feature = pose_feature.unsqueeze(0)

    next_character_velocity = pose_feature[:, :3]
    next_character_angular_velocity = pose_feature[:, 3:6]
    next_root_position = pose_feature[:, 6:9]
    next_root_rotation = pose_feature[:, 12:16]
    next_joint_rotations = pose_feature[:, 19:103]

    # character's current state
    curr_facing_dir = smpl.get_facing_directions(body_params[-1:])
    curr_position = smpl.get_character_positions(body_params[-1:])  # (1, 3)
    curr_rotation = smpl.get_character_rotations(curr_facing_dir)  # (1, 4)

    # get character's position and rotation at the next frame
    next_position = curr_position + to_world(next_character_velocity * dt, curr_rotation)
    next_rotation = quaternion_multiply(
        axis_angle_to_quaternion(next_character_angular_velocity * dt),
        curr_rotation,
    )

    # get character's root position and rotation based on the character's next position and rotation
    next_root_position = to_world(next_root_position, next_position, next_rotation)
    next_root_rotation = to_world(next_root_rotation, next_rotation)

    init_root_position = torch.tensor(smpl.INITIAL_ROOT_POSITION_MALE, dtype=torch.float32, device=body_params.device)
    next_body_params = torch.cat(
        [
            next_root_position - init_root_position,
            next_root_rotation,
            next_joint_rotations,
        ],
        dim=1,
    )
    return next_body_params


def replace_upper_body(src_pose_feature, rpl_pose_feature):
    character_velocity_src = src_pose_feature[:3]
    character_angular_velocity_src = src_pose_feature[3:6]
    root_position_src = src_pose_feature[6:9]
    root_velocity_src = src_pose_feature[9:12]
    root_rotation_src = src_pose_feature[12:16]
    root_angular_velocity_src = src_pose_feature[16:19]
    joint_rotations_src = src_pose_feature[19:103]
    joint_rotations_rpl = rpl_pose_feature[19:103]
    joint_angular_velocities_src = src_pose_feature[103:]
    joint_angular_velocities_rpl = rpl_pose_feature[103:]

    # replace the body pose of the upper body
    # joint_rotations_src[8:12] = joint_rotations_rpl[8:12]  # pelvis -> spine1
    # joint_angular_velocities_src[6:9] = joint_angular_velocities_rpl[6:9]
    # joint_rotations_src[20:24] = joint_rotations_rpl[20:24]  # spine1 -> spine2
    # joint_angular_velocities_src[15:18] = joint_angular_velocities_rpl[15:18]
    # joint_rotations_src[32:36] = joint_rotations_rpl[32:36]  # spine2 -> spine3
    # joint_angular_velocities_src[24:27] = joint_angular_velocities_rpl[24:27]
    joint_rotations_src[44:] = joint_rotations_rpl[44:]  # above spine3
    joint_angular_velocities_src[33:] = joint_angular_velocities_rpl[33:]

    pose_features = torch.cat(
        [
            character_velocity_src,  # :3
            character_angular_velocity_src,  # 3:6
            root_position_src,  # 6:9
            root_velocity_src,  # 9:12
            root_rotation_src,  # 12:16
            root_angular_velocity_src,  # 16:19
            joint_rotations_src,  # 19:103
            joint_angular_velocities_src,  # 103:166
        ],
    )
    assert pose_features.shape[-1] == 166, f"{pose_features.shape[-1]}"
    return pose_features


def to_tensor(body_params):
    """Convert the body parameters of dictionary format to tensor format.
    Rotations are converted to quaternion and hand poses are discarded

    :param body_params: body parameters in dictionary format
    :return: body parameters in torch.Tensor format
    """
    assert isinstance(body_params, dict)
    return torch.cat(
        [
            body_params["transl"],
            axis_angle_to_quaternion(body_params["global_orient"]),
            axis_angle_to_quaternion(body_params["body_pose"].reshape(-1, 21, 3)).reshape(-1, 84),
        ],
        dim=1,
    )


def to_dict(body_params):
    """Convert the body parameters in tnesor format to dictionary format, which can be directly consumed by SMPL model.
    Rotations are converted back to axis-angle format.

    :param body_params: body parameters in torch.Tensor format
    :return: body parameters in dictionary format
    """
    assert isinstance(body_params, torch.Tensor)
    return {
        "transl": body_params[:, :3],
        "global_orient": quaternion_to_axis_angle(body_params[:, 3:7]),
        "body_pose": quaternion_to_axis_angle(body_params[:, 7:].reshape(-1, 21, 4)).reshape(-1, 63),
    }


def to_local(features, *args):
    if features.shape[-1] == 4:
        character_rotations = args[0]
        return _to_local_coord_rotational(features, character_rotations)
    elif len(args) == 1:
        character_rotations = args[0]
        return _to_local_coord_directional(features, character_rotations)
    else:
        character_positions = args[0]
        character_rotations = args[1]
        return _to_local_coord_positional(features, character_positions, character_rotations)


def to_world(features, *args):
    if features.shape[-1] == 4:
        character_rotations = args[0]
        return _to_world_coord_rotational(features, character_rotations)
    elif len(args) == 1:
        character_rotations = args[0]
        return _to_world_coord_directional(features, character_rotations)
    else:
        character_positions = args[0]
        character_rotations = args[1]
        return _to_world_coord_positional(features, character_positions, character_rotations)


def _to_local_coord_positional(features, character_positions, character_rotations):
    if features.ndim > 2:
        n = features.ndim - character_positions.ndim
        character_positions = character_positions.view(-1, *(1,) * n, 3)
        character_rotations = character_rotations.view(-1, *(1,) * n, 4)

    localized_features = features - character_positions
    localized_features = quaternion_apply(
        quaternion_invert(character_rotations),
        localized_features,
    )
    return localized_features.squeeze(1)


def _to_local_coord_directional(features, character_rotations):
    if features.ndim > 2:
        n = features.ndim - character_rotations.ndim
        character_rotations = character_rotations.view(-1, *(1,) * n, 4)

    localized_features = quaternion_apply(
        quaternion_invert(character_rotations),
        features,
    )
    return localized_features.squeeze(1)


def _to_local_coord_rotational(features, character_rotations):
    assert features.shape[-1] == 4, "Features should be in quaternion format."
    if features.ndim > 2:
        n = features.ndim - character_rotations.ndim
        character_rotations = character_rotations.view(-1, *(1,) * n, 4)

    localized_features = quaternion_multiply(
        quaternion_invert(character_rotations),
        features,
    )
    return localized_features.squeeze(1)


def _to_world_coord_positional(features, character_positions, character_rotations):
    if features.ndim > 2:
        n = features.ndim - character_positions.ndim
        character_positions = character_positions.view(-1, *(1,) * n, 3)
        character_rotations = character_rotations.view(-1, *(1,) * n, 4)

    globalized_features = quaternion_apply(character_rotations, features) + character_positions
    return globalized_features.squeeze(1)


def _to_world_coord_directional(features, character_rotations):
    if features.ndim > 2:
        n = features.ndim - character_rotations.ndim
        character_rotations = character_rotations.view(-1, *(1,) * n, 4)

    globalized_features = quaternion_apply(character_rotations, features)
    return globalized_features.squeeze(1)


def _to_world_coord_rotational(features, character_rotations):
    assert features.shape[-1] == 4, "Features should be in quaternion format."
    if features.ndim > 2:
        n = features.ndim - character_rotations.ndim
        character_rotations = character_rotations.view(-1, *(1,) * n, 4)

    globalized_features = quaternion_multiply(character_rotations, features)
    return globalized_features.squeeze(1)


def _get_device(body_params):
    if isinstance(body_params, torch.Tensor):
        return body_params.device
    elif isinstance(body_params, dict):
        return body_params["transl"].device
    else:
        raise ValueError("Invalid body_params type")
