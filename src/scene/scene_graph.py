import itertools
import json
import re
from collections import defaultdict
from typing import List, Tuple

import networkx as nx

from ..type import Object
from . import relationships as sg


class SceneGraph:
    def __init__(self, scene_name, cfg, objects: List[Object], floor_level=0) -> None:
        self.cfg = cfg
        self.scene_name = scene_name
        self.boundary_offset = cfg.scene_graph.boundary_offset
        self.adjacent_threshold = cfg.scene_graph.adjacent_threshold
        self.close_threshold = cfg.scene_graph.close_threshold
        self.relationship_xy_threshold = cfg.scene_graph.relationship_xy_threshold
        self.relationship_z_threshold = cfg.scene_graph.relationship_z_threshold
        self.floor_level = floor_level
        self.objects = objects

        self._obj_finder = {f"{obj.label}_{obj.idx}": obj for obj in self.objects}
        self.G = nx.MultiDiGraph()
        for obj in self.objects:
            self.G.add_node(
                str(obj),
                id=obj.id,
                label=obj.label,
                aa_extents=obj.aa_extents,
                orientation=obj.orientation,
                has_orientation=obj.has_orientation,
                width=obj.width,
                depth=obj.depth,
                height=obj.height,
                center=obj.center,
                bbox=obj.bbox,
            )
        self.construct_edges()

    def construct_edges(self):
        # classify hierarchy levels based on the supporting relationships
        hangable_objs = set(self.objects)
        supported_objs = defaultdict(set)

        # objects supported by floor
        for obj in self.objects:
            if abs(obj.min_z - self.floor_level) < self.boundary_offset:
                supported_objs["floor"].add(obj)
                hangable_objs.discard(obj)

        # objects supported by other object
        for trg_obj, anchor_obj in itertools.product(self.objects, repeat=2):
            if trg_obj == anchor_obj:
                continue
            if trg_obj in supported_objs["floor"]:
                continue

            rel = sg.get_in_contact_vertical_relationship(trg_obj, anchor_obj)
            if rel is not None:
                hangable_objs.discard(trg_obj)
                if rel == "on":
                    supported_objs[anchor_obj].add(trg_obj)

                trg_node = str(trg_obj)
                anchor_node = str(anchor_obj)
                self.G.add_edge(trg_node, anchor_node, label=rel)

        # get non-contact vertical relationships for hangable objects
        for hangable_obj, grounded_obj in itertools.product(hangable_objs, supported_objs["floor"]):
            rel = sg.get_non_contact_vertical_relationships(hangable_obj, grounded_obj)
            if rel is not None:
                trg_node = str(hangable_obj)
                anchor_node = str(grounded_obj)
                self.G.add_edge(trg_node, anchor_node, label=rel)

            rel = sg.get_non_contact_vertical_relationships(grounded_obj, hangable_obj)
            if rel is not None:
                trg_node = str(grounded_obj)
                anchor_node = str(hangable_obj)
                self.G.add_edge(trg_node, anchor_node, label=rel)

        # get horizontal relationships among the objects with same hierarchy
        for objs_with_same_hierarchy in supported_objs.values():
            for trg_obj, anchor_obj in itertools.product(objs_with_same_hierarchy, repeat=2):
                if trg_obj == anchor_obj:
                    continue

                z_distance = abs(trg_obj.min_z - anchor_obj.min_z)
                if z_distance > self.relationship_z_threshold:
                    continue

                xy_distance = sg.get_distance_between_surfaces(trg_obj, anchor_obj)
                if xy_distance > self.relationship_xy_threshold:
                    continue

                rel = sg.get_directional_relationship(trg_obj, anchor_obj)
                if rel:
                    trg_node = str(trg_obj)
                    anchor_node = str(anchor_obj)
                    self.G.add_edge(trg_node, anchor_node, label=rel)

                rel = sg.get_distance_relationship(
                    trg_obj,
                    anchor_obj,
                    adjacent_threshold=self.adjacent_threshold,
                    close_threshold=self.close_threshold,
                )
                if rel:
                    trg_node = str(trg_obj)
                    anchor_node = str(anchor_obj)
                    self.G.add_edge(trg_node, anchor_node, label=rel)

    def get_relationships(self, obj: Object) -> List[Tuple[Object, Object, str]]:
        if isinstance(obj, str):
            obj = self._obj_finder[obj]

        node = str(obj)
        relationships = []
        for src, trg, key, data in self.G.edges(node, data=True, keys=True):
            src_obj = self._obj_finder[src]
            trg_obj = self._obj_finder[trg]
            relationships.append((src_obj, trg_obj, data["label"]))
        return relationships

    def to_text(
        self,
        use_obj_positions=True,
        use_obj_orientation=False,
    ):
        texts = {}
        for node, data in self.G.nodes(data=True):
            texts[node] = {}
            if use_obj_positions:
                texts[node]["position"] = _round(data["center"])
                texts[node]["size"] = _round(data["aa_extents"])
                texts[node]["volume"] = round(data["width"] * data["depth"] * data["height"], 2)
            if use_obj_orientation and data["has_orientation"]:
                texts[node]["orientation"] = _trunc(data["orientation"])
            texts[node]["relationships"] = []
            for src, dest, key in self.G.edges(node, keys=True):
                edge = self.G.edges[src, dest, key]
                label = f"{edge['label']} {dest}"
                texts[node]["relationships"].append(label)
            if not texts[node]["relationships"]:
                del texts[node]["relationships"]

        return json.dumps(texts, cls=CustomEncoder)


class CustomEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(CustomEncoder, self).__init__(*args, **kwargs)
        self.original_encoder = json.JSONEncoder(indent=4)

    def encode(self, o):
        result = self.original_encoder.encode(o)
        result = re.sub(
            r"\[\s+(.+?)\s+\]",
            self._to_single_line,
            result,
            flags=re.DOTALL,
        )
        return result

    def _to_single_line(self, m):
        # if the list contains strings, don't put it on a single line
        if '"' in m.group(1):
            return m.group(0)
        else:  # otherwise, put it on a single line
            ret = m.group(1).replace("\n", "")
            ret = ret.replace("-0.0", "0.0")
            ret = re.sub(r"\s+", " ", ret)
            return "[" + ret + "]"


def _round(x, precision=2):
    return [round(x, precision) for x in x.tolist()]


def _trunc(x, precision=2):
    return [int(x * 10**precision) / 10**precision for x in x.tolist()]
