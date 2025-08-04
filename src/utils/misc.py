from __future__ import annotations

import base64
import json
import logging
import re

import numpy as np
import torch
from omegaconf import OmegaConf, read_write

# from yacs.config import CfgNode as CN
_LOGGERS = {}


class CustomEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(CustomEncoder, self).__init__(*args, **kwargs)
        self.original_encoder = json.JSONEncoder(indent=4)

    def encode(self, o):
        result = self.original_encoder.encode(o)
        result = re.sub(
            r'"(characters|position|orientation|target_action|relationships|sampled_position)": \[\s+(.+?)\s+\]',
            self._to_single_line,
            result,
            flags=re.DOTALL,
        )
        return result

    def _to_single_line(self, m):
        key = m.group(1)
        values = m.group(2).replace("\n", "")
        values = re.sub(r"\s+", " ", values)
        return f'"{key}": [{values}]'


def load_cfg(
    cfg_path: str = "./configs/default.yaml",
    read_only: bool = True,
    struct: bool = True,
):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, read_only)  # make the config read-only
    OmegaConf.set_struct(cfg, struct)  # prevent from adding new keys

    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli)
    with read_write(cfg):
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def get_console_logger(name: str, level: str = "INFO"):
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger


def canonicalize_situation(position, anchor_position, anchor_orientation):
    """
    Canonicalizes the situation by aligning the anchor's front direction with the +y-axis and its center with the origin.

    Args:
        position (numpy.ndarray): The position of the target object.
        anchor_position (numpy.ndarray): The position of the anchor object.
        anchor_orientation (numpy.ndarray): The orientation of the anchor object.

    Returns:
        numpy.ndarray: The canonicalized position of the target object.
    """
    # calculate the rotation to align the target object's front direction with the +y-axis
    theta = np.arctan2(anchor_orientation[1], anchor_orientation[0])
    rotation_angle = np.pi / 2 - theta
    R = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    # canonicalize the situation (trg_center is at the origin, trg_front_dir is aligned with the +y-axis)
    position = R @ (position - anchor_position)
    return position


def add_z(points, z=0.0):
    assert points.shape[-1] == 2
    if isinstance(points, np.ndarray):
        return np.concatenate([points, np.full((*points.shape[:-1], 1), z)], axis=-1)
    else:
        return torch.cat(
            [
                points,
                torch.full((*points.shape[:-1], 1), z, dtype=points.dtype, device=points.device),
            ],
            dim=-1,
        )


def calculate_rotated_direction(initial, target, threshold_angle):
    """
    Calculate a new direction vector that does not rotate further than the threshold angle from the initial direction.
    This function assumes that the z-component of the input vectors is 0.
    """
    assert initial.shape == (1, 3)
    assert target.shape == (1, 3)
    device = initial.device

    initial = initial[0, :2]
    initial /= torch.norm(initial)
    target = target[0, :2]
    target /= torch.norm(target)
    threshold_angle = torch.tensor(threshold_angle, dtype=torch.float32, device=device)

    # calculate the cosine of the angle between the vectors
    cos_theta = (initial * target).sum()
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # calculate the angle between the vectors in radians
    angle = torch.acos(cos_theta)
    cross_product = initial[0] * target[1] - initial[1] * target[0]
    if cross_product < 0:
        angle = -angle

    if abs(angle) <= threshold_angle:
        return add_z(target, z=0).reshape(1, 3)

    if angle > 0:
        new_angle = threshold_angle
    else:
        new_angle = -threshold_angle

    # Calculate the new direction vector using the rotation matrix
    cos_new_angle = torch.cos(new_angle)
    sin_new_angle = torch.sin(new_angle)
    rotation_matrix = torch.tensor(
        [
            [cos_new_angle, -sin_new_angle],
            [sin_new_angle, cos_new_angle],
        ],
        device=device,
        dtype=torch.float32,
    ).reshape(2, 2)

    new_direction = rotation_matrix @ initial
    new_direction /= torch.norm(new_direction)
    return add_z(new_direction, z=0).reshape(1, 3)


# https://stackoverflow.com/a/18994296/15574032
def closest_distance_between_lines(
    a0,
    a1,
    b0,
    b1,
    clamp_all=False,
    clamp_a0=False,
    clamp_a1=False,
    clamp_b0=False,
    clamp_b1=False,
):
    """Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance
    """

    # If clampAll=True, set all clamps to True
    if clamp_all:
        clamp_a0 = True
        clamp_a1 = True
        clamp_b0 = True
        clamp_b1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clamp_a0 or clamp_a1 or clamp_b0 or clamp_b1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clamp_a0 and clamp_b1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clamp_a1 and clamp_b0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = b0 - a0
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clamp_a0 or clamp_a1 or clamp_b0 or clamp_b1:
        if clamp_a0 and t0 < 0:
            pA = a0
        elif clamp_a1 and t0 > magA:
            pA = a1

        if clamp_b0 and t1 < 0:
            pB = b0
        elif clamp_b1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clamp_a0 and t0 < 0) or (clamp_a1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clamp_b0 and dot < 0:
                dot = 0
            elif clamp_b1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clamp_b0 and t1 < 0) or (clamp_b1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clamp_a0 and dot < 0:
                dot = 0
            elif clamp_a1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
