import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import trimesh
import trimesh.creation

from ..type import Object
from ..utils import misc


class GridMap:
    def __init__(
        self,
        scene_name: str,
        scene_mesh: trimesh.Trimesh,
        objects: List[Object],
        cfg,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.scene_name = scene_name
        self.obj_convex_hulls = [obj.mesh.convex_hull for obj in objects]
        self.scene_mesh = trimesh.util.concatenate([scene_mesh] + self.obj_convex_hulls)
        self.bbox = self._load_scene_bounds(self.scene_name)
        try:
            self.update_grid_map(*self._load_grid_map_settings(self.scene_name))
        except FileNotFoundError:
            pass  # grid map will be updated later

    def __getitem__(self, idx):
        if not (isinstance(idx, (tuple, np.ndarray)) and len(idx) == 2):
            raise TypeError("Invalid argument type")
        x, y = idx
        return self.grid_points[x][y]

    def update_grid_map(
        self,
        grid_step,
        x_margin,
        y_margin,
        z_margin,
        cylinder_radius,
        cylinder_height,
        user_modifications=None,
    ):
        self.grid_step = grid_step
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.z_margin = z_margin
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        self.cylinder_min_z = self.bbox[0, 2] + self.z_margin
        (
            self.template_ray_origins,
            self.template_ray_directions,
        ) = self._create_template_rays(self.cylinder_radius, self.cylinder_height, self.cylinder_min_z)
        self.num_template_rays = len(self.template_ray_origins)

        min_x, min_y = self.bbox[0, :2]
        max_x, max_y = self.bbox[1, :2]
        X, Y, Z = torch.meshgrid(
            torch.arange(
                min_x + self.x_margin,
                max_x - self.x_margin,
                self.grid_step,
                device=self.device,
            ),
            torch.arange(
                min_y + self.y_margin,
                max_y - self.y_margin,
                self.grid_step,
                device=self.device,
            ),
            torch.tensor(0.0, device=self.device),
            indexing="ij",
        )

        self.num_x, self.num_y, _ = X.shape
        self.grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        self.num_grid_points = len(self.grid_points)

        origins = self.template_ray_origins + self.grid_points[:, None].cpu().numpy()
        origins = origins.reshape(-1, 3)
        directions = np.tile(self.template_ray_directions, (self.num_grid_points, 1, 1))
        directions = directions.reshape(-1, 3)
        _, ray_idx, intersections = self.scene_mesh.ray.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=True,
        )
        distances = np.full((self.num_grid_points * self.num_template_rays,), np.inf)
        distances[ray_idx] = np.linalg.norm(intersections - origins[ray_idx], axis=1)
        distances = distances.reshape(self.num_grid_points, -1)

        contained = np.full((self.num_grid_points,), False, dtype=bool)
        for convex_hull in self.obj_convex_hulls:
            contained |= convex_hull.contains(origins).reshape(self.num_grid_points, -1).any(axis=-1)
        intersected = (distances <= cylinder_radius).any(axis=-1)

        self.occupied = contained | intersected
        self.occupied = self.occupied.reshape(self.num_x, self.num_y)
        self.grid_points = self.grid_points.reshape(self.num_x, self.num_y, 3).cpu().numpy()

        self._apply_user_modifications(user_modifications)

    def _create_template_rays(
        self,
        cylinder_radius,
        cylinder_height,
        cylinder_min_z,
        cylinder_sections=8,
        num_interpolate_steps=3,
    ):
        cylinder = trimesh.creation.cylinder(
            radius=cylinder_radius,
            height=cylinder_height,
            sections=cylinder_sections,
        )
        cylinder.apply_translation([0, 0, cylinder_height / 2 + cylinder_min_z])  # make bottom points lie in z = min_z

        bottom_points = np.concatenate(
            [
                cylinder.vertices[1:2],
                cylinder.vertices[:1],
                cylinder.vertices[4::2],
            ],
            axis=0,
        )
        top_points = np.concatenate(
            [
                cylinder.vertices[2:4],
                cylinder.vertices[5::2],
            ],
            axis=0,
        )
        surface_points = (
            bottom_points + (top_points - bottom_points) * np.linspace(0.0, 1.0, num_interpolate_steps)[:, None, None]
        )
        origins = np.mean(surface_points, axis=1, keepdims=True)
        origins = np.repeat(origins, surface_points.shape[1], axis=1)
        directions = surface_points - origins
        return (
            origins.reshape(-1, 3),
            directions.reshape(-1, 3),
        )

    def _apply_user_modifications(self, user_modifications=None):
        if user_modifications is None:
            return

        for x, y in user_modifications["occupied"]:
            self.occupied[x, y] = True
        for x, y in user_modifications["unoccupied"]:
            self.occupied[x, y] = False

    def is_penetration_occured(self, position):
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        if position.shape[-1] == 2:
            position = misc.add_z(position, z=0)

        origins = self.template_ray_origins + position
        directions = self.template_ray_directions
        for convex_hull in self.obj_convex_hulls:
            if convex_hull.contains(origins).any():
                return True

        _, ray_idx, intersections = self.scene_mesh.ray.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=True,
        )
        distances = np.full((self.num_template_rays,), np.inf)
        distances[ray_idx] = np.linalg.norm(intersections - origins[ray_idx], axis=1)
        if (distances <= self.cylinder_radius).any():
            return True
        return False

    def is_off_boundary(self, position):
        if not isinstance(position, np.ndarray):
            position = position.cpu().numpy()
        if position.ndim == 2:
            position = position[0]

        min_bound = self.bbox[0, :2] + np.array([self.x_margin, self.y_margin])
        max_bound = self.bbox[1, :2] - np.array([self.x_margin, self.y_margin])
        if np.any(position[:2] < min_bound) or np.any(position[:2] > max_bound):
            return True
        return False

    def get_closest_grid_position(self, position, exclude_occupied=True, return_tensor=False):
        if not isinstance(position, np.ndarray):
            position = position.cpu().numpy()

        distances = np.linalg.norm(self.grid_points - position, axis=-1)
        if exclude_occupied:
            distances[self.occupied] = np.inf
        idx = np.argmin(distances)
        x, y = idx // self.num_y, idx % self.num_y
        if return_tensor:
            return torch.tensor(
                self.grid_points[x, y],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)
        return int(x), int(y)

    def get_random_position(self, return_tensor=False):
        unoccupied_grid_points = self.grid_points[~self.occupied]
        position = random.choice(unoccupied_grid_points)
        if return_tensor:
            return torch.tensor(position, dtype=torch.float32, device=self.device)
        return position

    def get_grid_points(self, grid_path, required_num_points=None):
        if required_num_points is not None:
            grid_path = grid_path[:required_num_points]
            grid_path += [(0, 0)] * (required_num_points - len(grid_path))

        indices = (
            np.array([p[0] for p in grid_path]),
            np.array([p[1] for p in grid_path]),
        )
        return self.grid_points[indices]  # (N, 3)

    def _load_grid_map_settings(self, scene_name):
        grid_map_setting_file = Path(f"./configs/scenes/{scene_name}/grid_map.json")

        # create default scene setting file if not exists
        if not grid_map_setting_file.exists():
            default_grid_map_setting = {
                "grid_step": 0.3,
                "x_margin": 0.1,
                "y_margin": 0.1,
                "z_margin": 0.1,
                "cylinder_radius": 0.1,
                "cylinder_height": 0.7,
                "user_modifications": {
                    "occupied": [],
                    "unoccupied": [],
                },
            }
            with open(grid_map_setting_file, "w") as f:
                json.dump(default_grid_map_setting, f, indent=2)

        # load scene settings
        with open(grid_map_setting_file, "r") as f:
            grid_map_settings = json.load(f)

        user_modifications = grid_map_settings.get("user_modifications", None)
        if user_modifications is None:
            return (
                grid_map_settings["grid_step"],
                grid_map_settings["x_margin"],
                grid_map_settings["y_margin"],
                grid_map_settings["z_margin"],
                grid_map_settings["cylinder_radius"],
                grid_map_settings["cylinder_height"],
                None,
            )

        user_modifications["occupied"] = set(tuple(p) for p in user_modifications["occupied"])
        user_modifications["unoccupied"] = set(tuple(p) for p in user_modifications["unoccupied"])
        return (
            grid_map_settings["grid_step"],
            grid_map_settings["x_margin"],
            grid_map_settings["y_margin"],
            grid_map_settings["z_margin"],
            grid_map_settings["cylinder_radius"],
            grid_map_settings["cylinder_height"],
            user_modifications,
        )

    def _load_scene_bounds(self, scene_name):
        transform_data_file = Path(f"./configs/scenes/{scene_name}/scene.json")
        with open(transform_data_file, "r") as f:
            data = json.load(f)
            bbox = np.array(data["bbox"], dtype=np.float32).reshape(2, 3)
        return bbox
