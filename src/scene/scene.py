import json
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from tqdm import tqdm

from ..type import Object
from .grid_map import GridMap
from .scene_graph import SceneGraph


class Scene:
    def __init__(self, scene_name, cfg):
        self.cfg = cfg
        self.scene_name = scene_name
        self.objects = self._get_objects(scene_name)
        self.scene_mesh = self._get_scene_mesh(self.objects)
        self.grid_map = GridMap(scene_name, self.scene_mesh, self.objects, cfg)
        self.scene_graph = SceneGraph(scene_name, self.cfg, self.objects)

    def _get_objects(self, scene_name) -> List[Object]:
        meshes = self._load_object_meshes(scene_name)
        annotations = self._load_annotations(scene_name)

        objects = []
        for id_, mesh in enumerate(meshes):
            anno = annotations[id_]
            objects.append(
                Object(
                    id=anno["id"],
                    label=anno["label"],
                    idx=anno["idx"],
                    aa_extents=np.array(anno["aa_extents"], dtype=np.float32),
                    orientation=np.array(anno["orientation"], dtype=np.float32),
                    has_orientation=anno["has_orientation"],
                    width=anno["width"],
                    depth=anno["depth"],
                    height=anno["height"],
                    center=np.array(anno["center"], dtype=np.float32),
                    bbox=np.array(anno["bbox"], dtype=np.float32),
                    mesh=mesh,
                    sit_offset_horizontal=anno["sit_offset_horizontal"],
                    sit_offset_vertical=anno["sit_offset_vertical"],
                )
            )
        return sorted(objects, key=lambda obj: obj.label)

    def _get_scene_mesh(self, objects: List[Object]):
        meshes = [obj.mesh.copy() for obj in objects]
        for mesh in meshes:
            mesh.visual = mesh.visual.to_color()
        return trimesh.util.concatenate(meshes)

    def _load_object_meshes(self, scene_name):
        scene_cfg_file = Path(f"./configs/scenes/{scene_name}/scene_cfg.json")
        scene_cfg = json.load(scene_cfg_file.open())
        scene_transform = self._load_scene_transform(scene_name)

        meshes = []
        for id_, (_, data) in tqdm(enumerate(scene_cfg.items()), desc=f"Loading {scene_name}", total=len(scene_cfg)):
            assert id_ == data["id"], f"ID mismatch: {id_} != {data['id']}"
            model_id = data["model_id"]
            location = data["location"]
            rotation = data["rotation"]
            scale = data["scale"]

            # load mesh and apply transformations
            mesh = self._load_obj_mesh(model_id)
            transform = trimesh.transformations.compose_matrix(
                translate=location,
                angles=rotation,
                scale=scale,
            )
            mesh.apply_transform(transform)
            mesh.apply_transform(scene_transform)

            meshes.append(mesh)

        return meshes

    def _load_scene_transform(self, scene_name):
        transform_data_file = Path(f"./configs/scenes/{scene_name}/scene.json")
        with open(transform_data_file, "r") as f:
            data = json.load(f)
            transform = np.array(data["transform"], dtype=np.float32).reshape(4, 4)
        return transform

    def _load_obj_mesh(self, model_id: str):
        obj_file = f"./data/HSSD/objects/{model_id[0]}/{model_id}.glb"
        obj_mesh = trimesh.load(obj_file)
        obj_mesh = obj_mesh.to_geometry()
        obj_mesh = obj_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]),
        )  # y-up to z-up
        return obj_mesh

    def _load_annotations(self, scene_name):
        annotation_file = Path(f"./configs/scenes/{scene_name}/objects.json")
        with annotation_file.open() as f:
            annotations = json.load(f)

        return {anno["id"]: {k: v for k, v in anno.items()} for anno in annotations}
