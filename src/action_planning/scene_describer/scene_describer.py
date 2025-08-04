from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN

import src.scene.relationships as sg
import src.utils.misc as misc
from src.action_planning.llm_client import LLMClient
from src.scene.scene_graph import SceneGraph

load_dotenv()


class SceneDescriber:
    def __init__(self, cfg) -> None:
        self.client = LLMClient(
            provider=cfg.scene_describer.provider,
            model=cfg.scene_describer.model,
        )
        self.cfg = cfg
        self.provider = cfg.scene_describer.provider
        self.model = cfg.scene_describer.model
        self.temperature = cfg.scene_describer.temperature

        self.base_dir = Path(__file__).parent
        self.system_messages_dir = self.base_dir / "system_messages"
        self.examples_dir = self.base_dir / "examples"
        self.example_files = sorted(self.examples_dir.glob("*.yaml"))
        self.log_basedir = Path("./logs/scene_describer")
        self.logger = misc.get_console_logger(self.__class__.__name__, level=cfg.logging_level)

        # for object clustering
        self.use_clustering = cfg.scene_describer.use_clustering
        self.eps = cfg.scene_describer.eps
        self.min_samples = cfg.scene_describer.min_samples

        # for ablation study
        self.description_type = self.cfg.scene_describer.description_type
        assert self.description_type in [
            "narrative",
            "narrative_vlm",
            "object_list",
            "scene_graph",
            "scene_image",
        ]

        # for visual language model
        self.use_vlm = self.description_type == "narrative_vlm"
        if self.use_vlm:
            self.use_clustering = False

    def get_scene_description(self, scene_name, scene_graph: SceneGraph, with_usage=False):
        description = self.get_pregenerated_scene_description(scene_name)
        if description is not None:
            if with_usage:
                return description, 0, 0
            return description

        if self.description_type == "object_list":
            content = str([str(obj) for obj in scene_graph.objects])
            if with_usage:
                return content, 0, 0
            return content
        if self.description_type == "scene_graph":
            content = scene_graph.to_text(
                use_obj_positions=True,
                use_obj_orientation=True,
            )
            if with_usage:
                return content, 0, 0
            return content
        if self.description_type == "scene_image":
            objects = scene_graph.objects
            content = "\n".join(
                f'- "{obj.label}_{obj.idx}": ({obj.center[0]:.1f}, {obj.center[1]:.1f})' for obj in objects
            )
            img_path = f"./configs/scenes/{scene_name}/top_view.png"
            if with_usage:
                return (content, img_path), 0, 0
            return (content, img_path)

        self.logger.info(f"Generating scene description for {scene_name}...")
        args, messages = self.create_api_request(scene_graph, return_with_messages=True)
        content, prompt_tokens, completion_tokens = self.client.get_response(with_usage=True, **args)
        self.log(messages, content, scene_name)

        if with_usage:
            return content, prompt_tokens, completion_tokens
        return content

    def get_pregenerated_scene_description(self, scene_name, idx=0):
        if self.description_type not in ["narrative", "narrative_vlm"]:
            return None
        if not self.cfg.scene_describer.use_pregenerated_description:
            return None

        description_dir = Path(
            f"./configs/scenes/{scene_name}/descriptions/{self.provider}-{self.model.split('/')[-1]}-{self.temperature}"
        )
        if self.use_vlm:
            description_dir = description_dir.with_name(description_dir.name + "-vlm")
        description_files = sorted(description_dir.glob("*.txt"))
        if not description_files:
            raise FileNotFoundError(f"No pregenerated scene description found in '{description_dir}'")

        description_file = description_files[idx]
        with open(description_file, "r") as f:
            description = f.read()

        self.logger.info(f"Using pregenerated scene description from '{description_file}'")
        return description

    def create_api_request(self, scene_graph: SceneGraph, return_with_messages=False):
        messages = []
        messages.append({"role": "system", "content": self._create_system_message()})
        if self.cfg.scene_describer.use_examples:
            for example_file in self.example_files:
                user_part, assistant_part = self._create_example_message(example_file)
                messages.append({"role": "user", "content": user_part})
                messages.append({"role": "assistant", "content": assistant_part})
        if self.use_vlm:
            messages.append({"role": "user", "content": self._create_query_message_vlm(scene_graph)})
        else:
            messages.append({"role": "user", "content": self._create_query_message(scene_graph)})

        args = {
            "model": self.cfg.scene_describer.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if return_with_messages:
            return args, messages
        return args

    def _create_system_message(self):
        assert self.description_type in ["narrative", "narrative_vlm"]
        if self.description_type == "narrative":
            system_message_file = self.system_messages_dir / "scene_describer.txt"
        elif self.description_type == "narrative_vlm":
            system_message_file = self.system_messages_dir / "scene_describer_vlm.txt"
        else:
            raise ValueError(f"Invalid description type: {self.description_type}")

        system_message = system_message_file.read_text().strip()
        return system_message

    def _create_example_message(self, example_file):
        example_file = Path(example_file)
        example_data = yaml.safe_load(example_file.read_text())
        scene_name = example_data["scene_name"]
        scene_graph = SceneGraph(scene_name, self.cfg, use_preset=True)

        if self.use_vlm:
            user_part = self._create_query_message_vlm(scene_graph)
        else:
            user_part = self._create_query_message(scene_graph)
        assistant_part = example_data["response"].strip()
        return user_part, assistant_part

    def _create_query_message(self, scene_graph: SceneGraph):
        scene_graph_str = scene_graph.to_text(use_obj_positions=True, use_obj_orientation=True)
        clustering_str = ""
        clusters = self.get_object_clusters(scene_graph)
        for i, cluster in enumerate(clusters):
            cluster_str = str([str(obj) for obj in cluster]).replace("'", '"')
            clustering_str += f"- Cluster {i}: {cluster_str}\n"

        query_message = "[Scene Graph]\n"
        query_message += f"{scene_graph_str}\n"
        if self.use_clustering:
            query_message += "\n"
            query_message += "[Object Clustering]\n"
            query_message += f"{clustering_str}\n"
        return query_message.strip()

    def log(self, messages, response, scene_name):
        log_id = datetime.now().strftime("%y%m%d-%H%M%S")
        log_dir = self.log_basedir / scene_name / log_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{log_id}_{scene_name}.txt"

        with open(log_file, "w") as f:
            for i, message in enumerate(messages):
                if i == 0:
                    f.write(f"{'='*30} System Message {'='*30}\n")
                    f.write(message["content"])
                    f.write("\n\n")
                elif i == len(messages) - 1:
                    f.write(f"{'='*30} Query Messages {'='*30}\n")
                    if self.use_vlm:
                        f.write(message["content"][0]["text"])
                    else:
                        f.write(message["content"])
                    f.write("\n\n")
                elif i % 2 == 1:
                    f.write(f"{'='*30} Example {(i-1)//2} Query {'='*30}\n")
                    if self.use_vlm:
                        f.write(message["content"][0]["text"])
                    else:
                        f.write(message["content"])
                    f.write("\n\n")
                else:
                    f.write(f"{'='*30} Example {(i-1)//2} Response {'='*30}\n")
                    f.write(message["content"])
                    f.write("\n\n")

            f.write(f"{'='*30} Response {'='*30}\n")
            f.write(response)

    def get_object_clusters(self, scene_graph: SceneGraph):
        objects = scene_graph.objects
        distance_matrix = np.zeros((len(objects), len(objects)))
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                distance_matrix[i, j] = 0 if i == j else sg.get_distance_between_obbs(obj1, obj2)

        labels = (
            DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric="precomputed",
            )
            .fit(distance_matrix)
            .labels_
        )

        clusters = {}
        for obj, label in zip(objects, labels):
            if label == -1:
                continue

            if label not in clusters:
                clusters[label] = []
            clusters[label].append(obj)

        clusters = [v for v in clusters.values()]
        clusters = sorted(clusters, key=lambda x: len(x))
        return clusters

    def _create_query_message_vlm(self, scene_graph: SceneGraph):
        # user_part should contain two elements
        # 1 : Text that describing image
        # 2 : Image file
        def _create_user_content(type: str, data: str):
            if type == "text":
                return {"type": type, type: data}
            elif type == "image_url":
                return {"type": type, type: {"url": f"data:image/jpeg;base64,{data}"}}
            else:
                return ValueError(f"There is no type '{type}', type should be 'text' or 'image_url'")

        scene_name = scene_graph.scene_name
        img_path = f"./configs/scenes/{scene_name}/top_view.png"

        # create text_part with object list
        objects = scene_graph.objects
        obj_list_str = "\n".join(
            f'- "{obj.label}_{obj.idx}": ({obj.center[0]:.1f}, {obj.center[1]:.1f})' for obj in objects
        )
        obj_list_str = f"[Object List]\n{obj_list_str}"
        text_part = _create_user_content("text", obj_list_str)

        # create image_part with image data
        img_data_str = misc.encode_image(img_path)
        image_part = _create_user_content("image_url", img_data_str)

        user_part = [text_part, image_part]
        return user_part
