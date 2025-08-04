"""
All code snippet in this file are borrowed from this article:
https://theorangeduck.com/page/spring-roll-call
"""

import torch

from src.utils.transform import (
    axis_angle_to_quaternion,
    quaternion_invert,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_to_axis_angle,
    standardize_quaternion,
)

# Constants
LN2 = float(torch.log(torch.tensor(2.0)))
PI = float(torch.pi)


def fast_negexp(x):
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)


def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * LN2) / (halflife + eps)


def damp_adjustment_exact_vector(g, halflife, dt, eps=1e-5):
    return g * (1.0 - fast_negexp((LN2 * dt) / (halflife + eps)))


def damp_adjustment_exact_quaternion(g, halflife, dt, eps=1e-5):
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=g.device).tile(g.shape[0], 1)  # fmt: skip
    return quaternion_slerp(identity_quat, g, 1.0 - fast_negexp((LN2 * dt) / (halflife + eps)))


def decay_spring_damper_vector(x, v, halflife: float, dt: float):
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x * y  # (B, 3)
    eydt = fast_negexp(y * dt)

    x_new = eydt * (x + j1 * dt)
    v_new = eydt * (v - j1 * y * dt)
    return x_new, v_new  # (B, 3)


def decay_spring_damper_quaternion(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0

    j0 = quaternion_to_axis_angle(x)  # (B, 3)
    j1 = v + j0 * y  # (B, 3)

    eydt = fast_negexp(y * dt)  # (B,)
    new_x = standardize_quaternion(axis_angle_to_quaternion(eydt * (j0 + j1 * dt)))  # (B, 4)
    new_v = eydt * (v - j1 * y * dt)  # (B, 3)
    return new_x, new_v


def spring_character_update(x, v, a, v_goal, halflife, dt):
    y = halflife_to_damping(halflife) / 2
    j0 = v - v_goal
    j1 = a + j0 * y
    eydt = fast_negexp(y * dt)

    x = eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y)) + (j1 / (y * y)) + j0 / y + v_goal * dt + x
    v = eydt * (j0 + j1 * dt) + v_goal
    a = eydt * (a - j1 * y * dt)
    return x, v, a


def simple_spring_damper_exact_quaternion(x, v, x_goal, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0

    j0 = quaternion_to_axis_angle((quaternion_multiply(x, quaternion_invert(x_goal))))
    j1 = v + j0 * y

    eydt = fast_negexp(y * dt)
    new_x = quaternion_multiply(axis_angle_to_quaternion(eydt * (j0 + j1 * dt)), x_goal)
    new_v = eydt * (v - j1 * y * dt)
    return new_x, new_v
