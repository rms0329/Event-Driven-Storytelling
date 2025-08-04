import itertools

import numpy as np

from ..type import Object
from ..utils import misc

# (relationship, anchor_label) -> new_relationship
# (relationship, target_label, anchor_label) -> new_relationship
_RELATIONSHIP_CHANGE_RULES = {
    ("inside", "table"): "on",
    ("inside", "desk"): "on",
    ("inside", "bed"): "on",
    ("inside", "sofa"): "on",
    ("inside", "couch"): "on",
    ("inside", "bench"): "on",
    ("inside", "kitchen_cabinet"): "on",
    ("inside", "sink", "cabinet"): "embedded in",
    ("inside", "shelf", "cabinet"): "embedded in",
    ("inside", "oven", "cabinet"): "embedded in",
    ("inside", "microwave", "cabinet"): "on",
    ("inside", "coffee_maker", "cabinet"): "on",
    ("inside", "coffee_machine", "cabinet"): "on",
    ("on", "radiator"): None,
}


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


def get_directional_relationship(
    target: Object,
    anchor: Object,
    offset=0.2,
    self_coverage_threshold=0.7,
    anchor_coverage_threshold=0.7,
    verbose=False,
):
    if not anchor.has_orientation:
        return None

    rel = _get_directional_relationship_by_center(target, anchor, offset=offset)
    if rel is not None:
        return rel

    rel = _get_directional_relationship_by_coverage(
        target,
        anchor,
        self_coverage_threshold=self_coverage_threshold,
        anchor_coverage_threshold=anchor_coverage_threshold,
        verbose=verbose,
    )
    return rel


def _get_directional_relationship_by_coverage(
    target: Object,
    anchor: Object,
    offset=0.0,
    self_coverage_threshold=0.7,
    anchor_coverage_threshold=0.7,
    verbose=False,
):
    if not anchor.has_orientation:
        return
    if verbose:
        print(f"\n\ntarget: {target}, anchor: {anchor}")

    w, h = anchor.width, anchor.depth
    corners = [canonicalize_situation(p, anchor.center[:2], anchor.orientation) for p in target.bbox[:4, :2]]
    surface_lines = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]

    p1, p2 = np.inf, -np.inf
    for line in surface_lines:
        if line[0][1] > h / 2 + offset and line[1][1] > h / 2 + offset:
            p1 = min(line[0][0], line[1][0], p1)
            p2 = max(line[0][0], line[1][0], p2)
    overlapped_len = min(p2, w / 2) - max(p1, -w / 2)
    self_coverage = overlapped_len / (p2 - p1)
    anchor_coverage = overlapped_len / w
    if verbose:
        print(f"  - front: self={self_coverage:.2f}, anchor={anchor_coverage:.2f}")
    if self_coverage >= self_coverage_threshold or anchor_coverage > anchor_coverage_threshold:
        return "in front of"

    p1, p2 = np.inf, -np.inf
    for line in surface_lines:
        if line[0][1] < -h / 2 - offset and line[1][1] < -h / 2 - offset:
            p1 = min(line[0][0], line[1][0], p1)
            p2 = max(line[0][0], line[1][0], p2)
    overlapped_len = min(p2, w / 2) - max(p1, -w / 2)
    self_coverage = overlapped_len / (p2 - p1)
    anchor_coverage = overlapped_len / w
    if verbose:
        print(f"  - behind: self={self_coverage:.2f}, anchor={anchor_coverage:.2f}")
    if self_coverage >= self_coverage_threshold or anchor_coverage > anchor_coverage_threshold:
        return "behind"

    p1, p2 = np.inf, -np.inf
    for line in surface_lines:
        if line[0][0] > w / 2 + offset and line[1][0] > w / 2 + offset:
            p1 = min(line[0][1], line[1][1], p1)
            p2 = max(line[0][1], line[1][1], p2)
    overlapped_len = min(p2, h / 2) - max(p1, -h / 2)
    self_coverage = overlapped_len / (p2 - p1)
    anchor_coverage = overlapped_len / h
    if verbose:
        print(f"  - right: self={self_coverage:.2f}, anchor={anchor_coverage:.2f}")
    if self_coverage >= self_coverage_threshold or anchor_coverage > anchor_coverage_threshold:
        return "to the right of"

    p1, p2 = np.inf, -np.inf
    for line in surface_lines:
        if line[0][0] < -w / 2 - offset and line[1][0] < -w / 2 - offset:
            p1 = min(line[0][1], line[1][1], p1)
            p2 = max(line[0][1], line[1][1], p2)
    overlapped_len = min(p2, h / 2) - max(p1, -h / 2)
    self_coverage = overlapped_len / (p2 - p1)
    anchor_coverage = overlapped_len / h
    if verbose:
        print(f"  - left: self={self_coverage:.2f}, anchor={anchor_coverage:.2f}")
    if self_coverage >= self_coverage_threshold or anchor_coverage > anchor_coverage_threshold:
        return "to the left of"

    return None


def _get_directional_relationship_by_center(target: Object, anchor: Object, offset=0.2):
    if not anchor.has_orientation:
        return

    target_center = target.center[:2]
    anchor_center = anchor.center[:2]
    anchor_orientation = anchor.orientation
    anchor_width = anchor.width
    anchor_depth = anchor.depth

    # find the relationship
    target_center = canonicalize_situation(target_center, anchor_center, anchor_orientation)
    x, y = target_center
    if x < -anchor_width / 2 and y <= anchor_depth / 2 + offset and y >= -anchor_depth / 2 - offset:
        return "to the left of"
    elif x > anchor_width / 2 and y <= anchor_depth / 2 + offset and y >= -anchor_depth / 2 - offset:
        return "to the right of"
    elif y > anchor_depth / 2 and x <= anchor_width / 2 + offset and x >= -anchor_width / 2 - offset:
        return "in front of"
    elif y < -anchor_depth / 2 and x <= anchor_width / 2 + offset and x >= -anchor_width / 2 - offset:
        return "behind"
    return None


def get_distance_between_surfaces(target: Object, anchor: Object):
    target_corners = target.bbox[:4].copy()
    target_corners[:, 2] = 0
    anchor_corners = anchor.bbox[:4].copy()
    anchor_corners[:, 2] = 0

    target_surface_lines = [
        (target_corners[0], target_corners[1]),
        (target_corners[1], target_corners[2]),
        (target_corners[2], target_corners[3]),
        (target_corners[3], target_corners[0]),
    ]
    anchor_surface_lines = [
        (anchor_corners[0], anchor_corners[1]),
        (anchor_corners[1], anchor_corners[2]),
        (anchor_corners[2], anchor_corners[3]),
        (anchor_corners[3], anchor_corners[0]),
    ]
    min_distance = np.inf
    for l1, l2 in itertools.product(target_surface_lines, anchor_surface_lines):
        _, _, distance = misc.closest_distance_between_lines(*l1, *l2, clamp_all=True)
        if distance < min_distance:
            min_distance = distance
    return min_distance


def get_distance_between_obbs(obj_1: Object, obj_2: Object):
    def _get_distance_to_surface_canonical(w, d, h, x, y, z):
        w2, d2, h2 = w / 2.0, d / 2.0, h / 2.0
        t_x = w2 / abs(x) if x != 0 else np.inf
        t_y = d2 / abs(y) if y != 0 else np.inf
        t_z = h2 / abs(z) if z != 0 else np.inf
        t = min(t_x, t_y, t_z)
        distance = t * np.sqrt(x**2 + y**2 + z**2)
        return distance

    distance = np.linalg.norm(obj_1.center - obj_2.center)

    # center of obj_1 in the obj_2's canonical space
    w, d, h = obj_2.width, obj_2.depth, obj_2.height
    x, y = canonicalize_situation(obj_1.center[:2], obj_2.center[:2], obj_2.orientation)
    z = obj_1.center[2] - obj_2.center[2]
    distance -= _get_distance_to_surface_canonical(w, d, h, x, y, z)
    if distance < 0:
        return 0

    # center of obj_2 in the obj_1's canonical space
    w, d, h = obj_1.width, obj_1.depth, obj_1.height
    x, y = canonicalize_situation(obj_2.center[:2], obj_1.center[:2], obj_1.orientation)
    z = obj_2.center[2] - obj_1.center[2]
    distance -= _get_distance_to_surface_canonical(w, d, h, x, y, z)
    if distance < 0:
        return 0

    return distance


def get_distance_relationship(
    target: Object,
    anchor: Object,
    adjacent_threshold=0.3,
    close_threshold=1.0,
):
    distance = get_distance_between_surfaces(target, anchor)
    if distance < adjacent_threshold:
        return "adjacent to"
    elif distance < close_threshold:
        return "close to"
    return None


def get_in_contact_vertical_relationship(
    target: Object,
    anchor: Object,
    horizontal_offset=0.2,
    vertical_offset=0.2,
):
    # check if the target object's center is within the anchor object
    # if not, we don't make vertical relationship
    x, y = canonicalize_situation(target.center[:2], anchor.center[:2], anchor.orientation)
    if (
        x < -anchor.width / 2 - horizontal_offset
        or x > anchor.width / 2 + horizontal_offset
        or y < -anchor.depth / 2 - horizontal_offset
        or y > anchor.depth / 2 + horizontal_offset
    ):
        return None

    relationship = None
    if abs(target.min_z - anchor.max_z) <= vertical_offset:
        relationship = "on"
    elif anchor.min_z < target.center[2] < anchor.max_z:
        relationship = "inside"
    elif target.max_z < anchor.max_z and target.min_z > anchor.min_z:
        relationship = "inside"

    relationship = _apply_relationship_change_rules(
        target.label,
        anchor.label,
        relationship,
    )
    return relationship


def get_non_contact_vertical_relationships(
    target: Object,
    anchor: Object,
    horizontal_offset=0.4,
    vertical_offset=0.0,
):
    # check if the target object's center is within the anchor object
    # if not, we don't make vertical relationship
    x, y = canonicalize_situation(target.center[:2], anchor.center[:2], anchor.orientation)
    if (
        x < -anchor.width / 2 - horizontal_offset
        or x > anchor.width / 2 + horizontal_offset
        or y < -anchor.depth / 2 - horizontal_offset
        or y > anchor.depth / 2 + horizontal_offset
    ):
        return None

    relationship = None
    if target.min_z > anchor.max_z + vertical_offset:
        relationship = "above"
    elif target.max_z < anchor.min_z - vertical_offset:
        relationship = "below"

    relationship = _apply_relationship_change_rules(
        target.label,
        anchor.label,
        relationship,
    )
    return relationship


def _apply_relationship_change_rules(target_label, anchor_label, relationship):
    if (relationship, target_label, anchor_label) in _RELATIONSHIP_CHANGE_RULES:
        return _RELATIONSHIP_CHANGE_RULES[relationship, target_label, anchor_label]
    elif (relationship, anchor_label) in _RELATIONSHIP_CHANGE_RULES:
        return _RELATIONSHIP_CHANGE_RULES[relationship, anchor_label]
    return relationship


def get_distance_from_boundary(position, anchor: Object, offset=0.0):
    assert position.ndim == 1 and position.shape[0] == 2, "position must be a 2D vector"
    anchor_center = anchor.center[:2]
    anchor_orientation = anchor.orientation
    anchor_width = anchor.width
    anchor_depth = anchor.depth

    position = canonicalize_situation(position, anchor_center, anchor_orientation)
    x, y = position
    if x < -anchor_width / 2 - offset:
        if y > anchor_depth / 2 + offset:
            return np.linalg.norm(position - np.array([-anchor_width / 2 - offset, anchor_depth / 2 + offset]))
        elif y < -anchor_depth / 2 - offset:
            return np.linalg.norm(position - np.array([-anchor_width / 2 - offset, -anchor_depth / 2 - offset]))
        else:
            return -anchor_width / 2 - offset - x
    elif x > anchor_width / 2 + offset:
        if y > anchor_depth / 2 + offset:
            return np.linalg.norm(position - np.array([anchor_width / 2 + offset, anchor_depth / 2 + offset]))
        elif y < -anchor_depth / 2 - offset:
            return np.linalg.norm(position - np.array([anchor_width / 2 + offset, -anchor_depth / 2 - offset]))
        else:
            return x - anchor_width / 2 - offset
    else:
        if y > anchor_depth / 2 + offset:
            return y - anchor_depth / 2 - offset
        elif y < -anchor_depth / 2 - offset:
            return -anchor_depth / 2 - offset - y
        else:
            return 0.0


def get_distance_from_interaction(position, anchor: Object, target_interaction="sit on", offset=0.2):
    assert position.ndim == 1 and position.shape[0] == 2, "position must be a 2D vector"
    assert target_interaction in ["sit on", "lie on"]
    anchor_center = anchor.center[:2]
    anchor_orientation = anchor.orientation
    anchor_width = anchor.width
    anchor_depth = anchor.depth

    position = misc.canonicalize_situation(position, anchor_center, anchor_orientation)
    x, y = position
    max_x = anchor_width / 2 - offset
    min_x = -anchor_width / 2 + offset
    max_y = anchor_depth / 2 - offset
    min_y = anchor_depth * anchor.sit_offset_horizontal  # we want the positions near the edge of anchor

    if x < min_x:
        if y > max_y:
            return np.linalg.norm(position - np.array([min_x, max_y]))
        elif y < min_y:
            return np.linalg.norm(position - np.array([min_x, min_y]))
        else:
            return min_x - x
    elif x > max_x:
        if y > max_y:
            return np.linalg.norm(position - np.array([max_x, max_y]))
        elif y < min_y:
            return np.linalg.norm(position - np.array([max_x, min_y]))
        else:
            return x - max_x
    else:
        if y > max_y:
            return y - max_y
        elif y < min_y:
            return min_y - y
        else:
            return 0.0


def get_distance_from_multi_obj_relationship(
    position,
    anchor_1: Object,
    anchor_2: Object,
    target_relationship,
    threshold_aligned=1.0,
    threshold_orthogonal=0.5,
):
    assert position.ndim == 1 and position.shape[0] == 2, "position must be a 2D vector"
    assert target_relationship in [
        "between",
        "aligned with",
    ]
    c1 = anchor_1.center[:2]
    c2 = anchor_2.center[:2]

    v = c2 - c1
    w = position - c1
    len_v = np.linalg.norm(v)

    t = np.dot(v, w) / (len_v * len_v + 1e-12)
    p_proj = c1 + t * v
    len_proj = np.linalg.norm(p_proj - position)

    if target_relationship == "between":
        if 0 <= t <= 1:
            dist_align = 0.0
        else:
            dist_align = len_v * (t - 1) if t > 1 else len_v * -t

        if len_proj < threshold_orthogonal:
            dist_orth = 0.0
        else:
            dist_orth = len_proj - threshold_orthogonal

        return dist_align + dist_orth

    elif target_relationship == "aligned with":
        if -threshold_aligned <= t <= 1 + threshold_aligned:
            dist_align = 0.0
        else:
            dist_align = (
                len_v * (t - (1 + threshold_aligned)) if t > 1 + threshold_aligned else len_v * (-threshold_aligned - t)
            )

        if len_proj < threshold_orthogonal:
            dist_orth = 0.0
        else:
            dist_orth = len_proj - threshold_orthogonal

        return dist_align + dist_orth

    else:
        raise ValueError(f"Unknown target relationship: {target_relationship}")


def get_distance_from_relationship(position, anchor: Object, target_relationship, offset=0.1):
    assert position.ndim == 1 and position.shape[0] == 2, "position must be a 2D vector"
    assert target_relationship in [
        "to the left of",
        "to the right of",
        "in front of",
        "behind",
    ]
    anchor_center = anchor.center[:2]
    anchor_orientation = anchor.orientation
    anchor_width = anchor.width
    anchor_depth = anchor.depth

    position = misc.canonicalize_situation(position, anchor_center, anchor_orientation)
    x, y = position

    if target_relationship == "to the left of":
        if x > -anchor_width / 2:
            if y > anchor_depth / 2:
                return np.linalg.norm(position - np.array([-anchor_width / 2, anchor_depth / 2]))
            elif y < -anchor_depth / 2:
                return np.linalg.norm(position - np.array([-anchor_width / 2, -anchor_depth / 2]))
            else:
                return x + anchor_width / 2
        else:
            if y > anchor_depth / 2 + offset:
                return y - anchor_depth / 2 - offset
            elif y < -anchor_depth / 2 - offset:
                return -anchor_depth / 2 - offset - y
            else:
                return 0.0
    elif target_relationship == "to the right of":
        if x < anchor_width / 2:
            if y > anchor_depth / 2:
                return np.linalg.norm(position - np.array([anchor_width / 2, anchor_depth / 2]))
            elif y < -anchor_depth / 2:
                return np.linalg.norm(position - np.array([anchor_width / 2, -anchor_depth / 2]))
            else:
                return anchor_width / 2 - x
        else:
            if y > anchor_depth / 2 + offset:
                return y - anchor_depth / 2 - offset
            elif y < -anchor_depth / 2 - offset:
                return -anchor_depth / 2 - offset - y
            else:
                return 0.0
    elif target_relationship == "in front of":
        if y < anchor_depth / 2:
            if x > anchor_width / 2:
                return np.linalg.norm(position - np.array([anchor_width / 2, anchor_depth / 2]))
            elif x < -anchor_width / 2:
                return np.linalg.norm(position - np.array([-anchor_width / 2, anchor_depth / 2]))
            else:
                return anchor_depth / 2 - y
        else:
            if x > anchor_width / 2 + offset:
                return x - anchor_width / 2 - offset
            elif x < -anchor_width / 2 - offset:
                return -anchor_width / 2 - offset - x
            else:
                return 0.0
    elif target_relationship == "behind":
        if y > -anchor_depth / 2:
            if x > anchor_width / 2:
                return np.linalg.norm(position - np.array([anchor_width / 2, -anchor_depth / 2]))
            elif x < -anchor_width / 2:
                return np.linalg.norm(position - np.array([-anchor_width / 2, -anchor_depth / 2]))
            else:
                return y + anchor_depth / 2
        else:
            if x > anchor_width / 2 + offset:
                return x - anchor_width / 2 - offset
            elif x < -anchor_width / 2 - offset:
                return -anchor_width / 2 - offset - x
            else:
                return 0.0
    else:
        raise ValueError(f"Unknown target relationship: {target_relationship}")
