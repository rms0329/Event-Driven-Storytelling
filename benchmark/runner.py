import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
from tqdm import tqdm

from src.action_planning.event_parser import get_event_parser
from src.action_planning.event_parser.position_sampler import PositionSampler
from src.action_planning.narrator import get_narrator
from src.action_planning.scene_describer.scene_describer import SceneDescriber
from src.scene.scene import Scene
from src.type import Character, Event
from src.utils import misc
from src.utils.misc import CustomEncoder

from .evaluator import Evaluator

_TESTCASE_DIR = Path("./benchmark/test_cases")
_GROUP_DIRS = {
    range(0, 5): _TESTCASE_DIR / "0_OA",
    range(10, 15): _TESTCASE_DIR / "1_RC",
    range(20, 25): _TESTCASE_DIR / "2_SS",
    range(30, 35): _TESTCASE_DIR / "3_OA_RC",
    range(40, 45): _TESTCASE_DIR / "4_OA_SS",
    range(50, 55): _TESTCASE_DIR / "5_RC_SS",
    range(60, 70): _TESTCASE_DIR / "6_PI",
}


def get_test_case_ids() -> list:
    """Returns a list of all test case IDs."""
    ids = []
    for range_ in _GROUP_DIRS:
        ids.extend(range_)
    return ids


def load_test_case(id_: int) -> dict:
    for range_, group_dir in _GROUP_DIRS.items():
        if id_ in range_:
            return json.loads((group_dir / f"testcase_{id_}.json").read_text())
    raise ValueError(f"Test case with ID {id_} not found.")


@dataclass
class TestMetadata:
    test_case_id: int
    distraction_level: int
    repeat_idx: int

    @classmethod
    def from_str(cls, metadata_str: str):
        m = re.match(r"tc(\d+)-d(\d+)-r(\d+)", metadata_str)
        if not m:
            raise ValueError(f"Invalid metadata string: {metadata_str}")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def __repr__(self):
        return f"tc{str(self.test_case_id).zfill(2)}-d{self.distraction_level}-r{self.repeat_idx}"

    def __hash__(self):
        return hash((self.test_case_id, self.distraction_level, self.repeat_idx))


class BenchmarkRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.evaluator = Evaluator()
        self.test_case_ids = cfg.evaluation.test_cases
        self.repeat_count = cfg.evaluation.repeat_count
        self.distraction_levels = cfg.evaluation.distraction_levels
        self.scene_description_type = cfg.scene_describer.description_type
        self.planning_mode = "event" if not cfg.narrator.disable_event_based_planning else "no_event"
        if cfg.event_parser.disable_lcps:
            self.planning_mode += "_no_lcps"

        self.run_id = datetime.now().strftime("%y%m%d-%H%M%S")
        self.logger = misc.get_console_logger("BenchmarkRunner", cfg.logging_level)
        self.log_base_dir = Path(
            f"./logs/benchmark/{cfg.evaluation.experiment_name}/"
            f"{self.run_id}-{self.scene_description_type}-{self.planning_mode}"
        )
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, self.log_base_dir / "config.yaml", resolve=True)

    def run_tests(self):
        test_case_ids_by_scene = self._group_test_cases_by_scene()
        for scene_name, test_case_ids in tqdm(test_case_ids_by_scene.items(), desc="Scenes"):
            if not test_case_ids:
                continue

            self._setup_scene(scene_name)
            for id_ in tqdm(test_case_ids, desc="Test cases"):
                test_case = load_test_case(id_)
                for distraction_level in tqdm(self.distraction_levels, desc="Distraction levels"):
                    max_distraction_level = self._get_max_distraction_level(test_case)
                    if distraction_level > max_distraction_level:
                        continue
                    for repeat_idx in tqdm(range(self.repeat_count), desc="Repeats", leave=False):
                        self.run_test(test_case, distraction_level, repeat_idx)

    def run_test(self, test_case, distraction_level, repeat_idx):
        self._setup_modules_and_scenario(test_case, distraction_level, repeat_idx)

        tags = test_case["tags"]
        metadata = TestMetadata(test_case["id"], distraction_level, repeat_idx)
        try:
            parsed_event, prompt_tokens, completion_tokens = self._run_planning()
            if "PI" in tags and not self.planning_mode.endswith("no_lcps"):
                for parsed_plan in parsed_event.values():
                    parsed_plan["sampled_position"] = self._sample_position(parsed_plan)

            passed = self.evaluator.evaluate_result(test_case, parsed_event)
            self._save_result(test_case, metadata, parsed_event, prompt_tokens, completion_tokens, passed)
        except Exception as e:
            self.logger.warning(f"Error occurred during: {metadata}")
            self._save_error_logs(metadata, e, traceback.format_exc())
            self._save_error_result(tags, metadata, e)

    def _run_planning(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0
        event, prompt_tokens, completion_tokens = self.narrator.generate_subsequent_plan(
            self.scene_description,
            self.available_action_labels,
            self.user_instruction,
            with_usage=True,
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        if event is None:  # when LLM wants to skip the turn
            return {}, total_prompt_tokens, total_completion_tokens
        if not self.narrator.event_based_planning:
            parsed_event = event  # already parsed
            return parsed_event, total_prompt_tokens, total_completion_tokens

        parsed_event, prompt_tokens, completion_tokens = self.event_parser.parse_event(
            event,
            self.scene_description,
            self.available_action_labels,
            self.narrator.previous_events[:-1],  # since the last event is the current event
            with_usage=True,
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        return parsed_event, total_prompt_tokens, total_completion_tokens

    def _group_test_cases_by_scene(self):
        test_case_ids = {"House": [], "Office": [], "Restaurant": []}
        for id_ in self.test_case_ids:
            test_case = load_test_case(id_)
            scene_name = test_case["scene_name"]
            test_case_ids[scene_name].append(id_)
        return test_case_ids

    def _setup_scene(self, scene_name):
        self.scene = Scene(scene_name, self.cfg)
        self.scene_name = self.scene.scene_name
        self.scene_mesh = self.scene.scene_mesh
        self.objects = self.scene.objects
        self.grid_map = self.scene.grid_map
        self.scene_graph = self.scene.scene_graph

    def _setup_modules_and_scenario(self, test_case, distraction_level, repeat_idx):
        self.user_instruction = test_case["user_instruction"]
        self.available_action_labels = test_case["action_labels"]

        # ----- setup characters and their current states -----
        self.characters: List[Character] = []
        for character in test_case["current_states"]:
            if character["type"] == "distractor" and character["distraction_level"] > distraction_level:
                continue
            character_ = Character(
                name=character["name"],
                state=character["state"],
                current_actions=[character["current_action"]],
                relationships=character["position"].copy() + character["orientation"].copy(),
            )
            if any(rel.startswith("sit on") for rel in character_.relationships):
                character_.current_actions.append("sit")
            if any(rel.startswith("lie on") for rel in character_.relationships):
                character_.current_actions.append("lie")
            self.characters.append(character_)

        # ----- setup scene description -----
        self.scene_describer = SceneDescriber(self.cfg)
        if self.scene_description_type in ["narrative", "narrative_vlm"]:
            self.scene_description = self.scene_describer.get_pregenerated_scene_description(
                self.scene_name, idx=repeat_idx
            )
        else:
            self.scene_description = self.scene_describer.get_scene_description(self.scene_name, self.scene_graph)

        # ----- setup narrator -----
        self.narrator = get_narrator(self.characters, self.objects, self.cfg)
        self.narrator.available_characters = [c for c in self.characters if c.state == "idle"]
        if test_case["allow_null_plan"]:
            self.narrator.additional_guidelines += "- You can skip the event generation by outputting an empty event that involves no characters in it."  # fmt:skip
            self.narrator.additional_guidelines += " This is useful when you need to wait for some characters to become idle before the next event."  # fmt:skip
            if self.planning_mode.startswith("no_event"):
                self.narrator.additional_guidelines = self.narrator.additional_guidelines.replace("event", "plan")
        self.narrator.log_id_prefix = f"repeat-{repeat_idx}"
        self.narrator.log_basedir = (
            self.log_base_dir / "narrator" / f"testcase-{test_case['id']}" / f"distraction-{distraction_level}"
        )
        for example_file in self.narrator.example_files[:]:
            if example_file.stem not in test_case["tags"]:
                self.narrator.example_files.remove(example_file)

        # ----- setup planning history (previous events) -----
        for plan in test_case["planning_history"]:
            if plan["type"] == "distractor" and plan["distraction_level"] > distraction_level:
                continue
            event_ = Event(
                involved_characters=[c for c in self.characters if c.name in plan["characters"]],
                activity=plan["activity"],
                state=plan["state"],
            )
            for c in self.characters:
                if c.name not in plan["characters"]:
                    continue
                if plan["state"] == "completed":
                    c.event = None
                else:
                    c.event = event_  # assign event to character (overwrite previous one)

            parsed_event_ = {}
            for parsed in plan["details"]:
                parsed_event_[parsed["character"]] = {}
                parsed_event_[parsed["character"]]["target_action"] = [parsed["target_action"]]
                parsed_event_[parsed["character"]]["relationships"] = (
                    parsed["position"].copy() + parsed["orientation"].copy()
                )
            event_.parsed_event = parsed_event_
            self.narrator.previous_events.append(event_)

        # ----- setup event parser -----
        self.event_parser = get_event_parser(self.characters, self.objects, self.cfg)
        self.event_parser.log_id_prefix = f"repeat-{repeat_idx}"
        self.event_parser.log_basedir = (
            self.log_base_dir / "event_parser" / f"testcase-{test_case['id']}" / f"distraction-{distraction_level}"
        )
        self.event_parser.code_executor.set_previous_events(self.narrator.previous_events)
        for example_file in self.event_parser.example_files[:]:
            if example_file.stem not in test_case["tags"]:
                self.event_parser.example_files.remove(example_file)

        # ----- setup position sampler (used for PI tags) -----
        self.position_sampler = PositionSampler(self.grid_map, self.cfg)
        self.position_sampler.update_anchors(self.characters, self.objects)

        # ----- setup evaluators -----
        self.evaluator.setup_evaluator(
            self.available_action_labels,
            self.characters,
            self.objects,
        )

    def _save_result(self, test_case, metadata, planning_result, prompt_tokens, completion_tokens, passed):
        result_dir = (
            self.log_base_dir
            / "results"
            / f"testcase-{metadata.test_case_id}"
            / f"distraction-{metadata.distraction_level}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{metadata}.json"
        with open(result_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "scene_name": test_case["scene_name"],
                        "tags": test_case["tags"],
                        "run_id": self.run_id,
                        "scene_description_type": self.scene_description_type,
                        "planning_mode": self.planning_mode,
                        "test_case_id": metadata.test_case_id,
                        "distraction_level": metadata.distraction_level,
                        "repeat_idx": metadata.repeat_idx,
                        "user_instruction": test_case["user_instruction"],
                        "expected_plans": test_case["expected_plans"],
                        "allow_null_plan": test_case["allow_null_plan"],
                        "planning_result": planning_result,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "error_occurred": False,
                        "error_message": None,
                        "passed": passed,
                    },
                    cls=CustomEncoder,
                )
            )

    def _save_error_result(self, tags, metadata, exception):
        result_dir = (
            self.log_base_dir
            / "results"
            / f"testcase-{metadata.test_case_id}"
            / f"distraction-{metadata.distraction_level}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{metadata}.json"
        with open(result_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "tags": tags,
                        "run_id": self.run_id,
                        "scene_description_type": self.scene_description_type,
                        "planning_mode": self.planning_mode,
                        "test_case_id": metadata.test_case_id,
                        "distraction_level": metadata.distraction_level,
                        "repeat_idx": metadata.repeat_idx,
                        "parsed_event": None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "error_occurred": True,
                        "error_message": f"{type(exception).__name__}: {str(exception)}",
                        "passed": False,
                    },
                    cls=CustomEncoder,
                )
            )

    def _save_error_logs(self, metadata, exception, stack_trace):
        error_log_dir = (
            self.log_base_dir
            / "errors"
            / f"testcase-{metadata.test_case_id}"
            / f"distraction-{metadata.distraction_level}"
        )
        error_log_dir.mkdir(parents=True, exist_ok=True)
        error_log_file = error_log_dir / f"{metadata}.txt"
        with open(error_log_file, "w") as f:
            f.write(f"{type(exception).__name__}: {str(exception)}\n")
            f.write(stack_trace)

    def _sample_position(self, parsed_plan):
        conditions = parsed_plan["relationships"]
        conditions = [c.replace("inside", "sit on") for c in conditions]
        conditions = [c.replace("interact with", "adjacent to") for c in conditions]
        sampled_position = (
            self.position_sampler.sample_position(
                conditions,
                positions_to_avoid=[],
                position_to_nearby=None,
                return_tensor=False,
            )
            .squeeze()[:2]
            .tolist()
        )
        return sampled_position

    def _get_max_distraction_level(self, test_case):
        max_distraction_level = 0
        for plan in test_case["planning_history"]:
            if plan["type"] == "distractor":
                max_distraction_level = max(max_distraction_level, plan["distraction_level"])
        for character in test_case["current_states"]:
            if character["type"] == "distractor":
                max_distraction_level = max(max_distraction_level, character["distraction_level"])
        return max_distraction_level
