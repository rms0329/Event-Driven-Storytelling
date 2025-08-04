import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

from src.action_planning.event_parser.code_executor import CodeExecutor
from src.action_planning.llm_client import LLMClient
from src.type import Character, Event, Object
from src.utils import misc

_VALID_RELATIONSHIPS = [
    "interact with",
    "adjacent to",
    "close to",
    "to the left of",
    "to the right of",
    "behind",
    "in front of",
    "look at",
    "sit on",
    "lie on",
]


class EventParser:
    def __init__(
        self,
        characters: List[Character],
        objects: List[Object],
        cfg,
    ) -> None:
        self.client = LLMClient(
            provider=cfg.event_parser.provider,
            model=cfg.event_parser.model,
        )
        self.cfg = cfg
        self.characters = characters
        self.character_names = set(character.name for character in characters)
        self.objects = objects
        self.object_labels = set(f"{obj.label}_{obj.idx}" for obj in objects)

        self.model = cfg.event_parser.model
        self.temperature = cfg.event_parser.temperature
        self.use_self_feedback = cfg.event_parser.use_self_feedback
        self.use_examples = cfg.event_parser.use_examples
        self.use_json_mode = cfg.event_parser.use_JSON_mode
        self.base_dir = Path(__file__).parent
        self.system_messages_dir = self.base_dir / "system_messages"
        self.system_message_file = self.system_messages_dir / "event_parser.txt"
        self.additional_guidelines = cfg.event_parser.additional_guidelines
        self.examples_dir = self.base_dir / "examples"
        self.example_files = sorted(self.examples_dir.glob("*.yaml"))
        self.log_basedir = Path("./logs/event_parser")
        self.log_id_prefix = None
        self.scene_descriptions_dir = Path("./src/action_planning/scene_describer/sample_descriptions")
        self.scene_description_type = cfg.scene_describer.description_type
        self.max_self_feedback_count = cfg.event_parser.max_self_feedback_count
        self.self_feedback_count = 0
        self.logger = misc.get_console_logger(self.__class__.__name__, level=cfg.logging_level)

        self.use_json_mode = False
        self.code_executor = CodeExecutor(characters, objects, cfg)

    def parse_event(
        self,
        event: Event,
        scene_description,
        available_action_labels,
        previous_events,
        feedback=None,
        with_usage=False,
        accumulated_prompt_tokens=0,
        accumulated_completion_tokens=0,
    ):
        self.logger.info(f"Parsing the event: {event}...")
        self.code_executor.set_previous_events(previous_events)

        response, prompt_tokens, completion_tokens = self._query_llm(
            event, scene_description, available_action_labels, previous_events, feedback
        )
        response = self._post_process_response(event, response)
        feedback = self._check_validity(event, response, available_action_labels)
        if feedback and self.self_feedback_count < self.max_self_feedback_count:
            self.logger.warning("Invalid response detected, retrying with feedback...")
            self.logger.warning(f"feedback: {feedback}")
            self.self_feedback_count += 1
            return self.parse_event(
                event,
                scene_description,
                available_action_labels,
                previous_events,
                feedback,
                with_usage,
                accumulated_prompt_tokens + prompt_tokens,
                accumulated_completion_tokens + completion_tokens,
            )

        self.self_feedback_count = 0
        if with_usage:
            return (
                response,
                prompt_tokens + accumulated_prompt_tokens,
                completion_tokens + accumulated_completion_tokens,
            )
        return response

    def _query_llm(
        self,
        event: Event,
        scene_description,
        available_action_labels,
        previous_events,
        feedback=None,
    ):
        args, messages = self.create_api_request(
            event,
            scene_description,
            available_action_labels,
            previous_events,
            feedback,
            return_with_messages=True,
        )
        response, prompt_tokens, completion_tokens = self.client.get_response(with_usage=True, **args)
        self.log(messages, response)
        return response, prompt_tokens, completion_tokens

    def create_api_request(
        self,
        event: Event,
        scene_description,
        available_action_labels,
        previous_events,
        feedback=None,
        return_with_messages=False,
    ):
        messages = []
        messages.append({"role": "system", "content": self._create_system_message()})
        if self.use_examples:
            for example_file in self.example_files:
                user_part, assistant_part = self._create_example_message(example_file)
                messages.append({"role": "user", "content": user_part})
                messages.append({"role": "assistant", "content": assistant_part})
        messages.append(
            {
                "role": "user",
                "content": self._create_query_message(
                    event, scene_description, available_action_labels, previous_events, feedback=feedback
                ),
            }
        )

        args = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.use_json_mode:
            args["response_format"] = {"type": "json_object"}

        if return_with_messages:
            return args, messages
        return args

    def _create_system_message(self):
        system_message = self.system_message_file.read_text().strip()
        if self.additional_guidelines:
            system_message += "\n"
            system_message += self.additional_guidelines.strip()
        return system_message

    def _create_example_message(self, example_file: Path):
        assert example_file.suffix == ".yaml"
        example_data = yaml.safe_load(example_file.read_text())

        # create user message part
        event_to_parse = Event(example_data["event"]["characters"], example_data["event"]["activity"])
        scene_description = (
            (self.scene_descriptions_dir / f"{example_data['scene_name']}_{self.scene_description_type}.txt")
            .read_text()
            .strip()
        )
        if self.scene_description_type == "scene_image":
            img_path = f"./configs/scenes/{example_data['scene_name']}/top_view.png"
            scene_description = (scene_description, img_path)
        available_action_labels = example_data["action_labels"]

        characters = []
        for character in example_data["characters"]:
            characters.append(
                Character(
                    name=character["name"],
                    state=character["state"],
                    current_actions=[character["current_action"]],
                    relationships=character["position"] + character["orientation"],
                )
            )

        previous_events = []
        for event in example_data["previous_events"]:
            previous_events.append(
                Event(
                    event["characters"],
                    event["activity"],
                    event["state"],
                )
            )
            parsed_event = {}
            for parsed in event["parsed"]:
                character = parsed["character"]
                parsed_event[character] = {}
                parsed_event[character]["target_action"] = [parsed["target_action"]]
                parsed_event[character]["relationships"] = parsed["position"] + parsed.get("orientation", [])
            previous_events[-1].parsed_event = parsed_event

        user_part = self._create_query_message(
            event_to_parse,
            scene_description,
            available_action_labels,
            previous_events,
            characters,
        )

        # create assistant message part
        example_data = yaml.safe_load(example_file.read_text())
        assistant_part = example_data["response"].strip()

        return user_part, assistant_part

    def _create_query_message(
        self,
        event: Event,
        scene_description: str,
        available_action_labels: List[str],
        previous_events: List[Event],
        characters: List[Character] = None,
        feedback=None,
    ):
        if characters is None:
            characters = self.characters

        scene_description_str = "[Scene Description]\n" + scene_description
        available_action_labels_str = "[Available Action Labels]\n" + str(available_action_labels)
        current_states_str = "[Current States of Characters]\n"
        for character in characters:
            current_states_str += f"- {character.name}:\n"
            current_states_str += f"  - state: '{character.state}'\n"
            if str(character.state) in ["idle", "interacting"]:
                current_states_str += (
                    f"  - current_action: '{self._get_main_action_label(character.current_actions)}'\n"
                )
                current_states_str += f"  - relationships: {character.relationships}\n"
            else:
                current_states_str += (
                    f"  - current_action: '{self._get_main_action_label(character.current_actions)}'\n"
                )
                current_states_str += f"  - target_action: '{self._get_main_action_label(character.target_actions)}'\n"
                current_states_str += f"  - target_relationships: {character.relationships}\n"
        event_str = f"[Event to Parse]\n{event}"
        feedback_str = f"[Feedback]\n{feedback if feedback else 'None'}"

        query_message = [
            scene_description_str,
            available_action_labels_str,
            current_states_str,
            event_str,
            feedback_str,
        ]
        query_message = map(str.strip, query_message)
        query_message = "\n\n".join(query_message).strip()
        return query_message

    def log(self, messages, response, log_id=None):
        if log_id is None:
            log_id = datetime.now().strftime("%y%m%d-%H%M%S")
        if self.log_id_prefix:
            log_id = f"{self.log_id_prefix}-{log_id}"
        log_dir = self.log_basedir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{log_id}.txt"

        with open(log_file, "w") as f:
            for i, message in enumerate(messages):
                if i == 0:
                    f.write(f"{'='*30} System Message {'='*30}\n")
                    f.write(message["content"])
                    f.write("\n\n")
                elif i == len(messages) - 1:
                    f.write(f"{'='*30} Query Messages {'='*30}\n")
                    f.write(message["content"])
                    f.write("\n\n")
                elif i % 2 == 1:
                    f.write(f"{'='*30} Example {(i-1)//2} Query {'='*30}\n")
                    f.write(message["content"])
                    f.write("\n\n")
                else:
                    f.write(f"{'='*30} Example {(i-1)//2} Response {'='*30}\n")
                    f.write(message["content"])
                    f.write("\n\n")

            f.write(f"{'='*30} Response {'='*30}\n")
            f.write(response)

    def _post_process_response(self, event: Event, response):
        parsed_event = self.code_executor.execute_response(response)

        # convert target_action to list
        for items in parsed_event.values():
            items["target_action"] = [items["target_action"]]

        # integrate position and orientation relationships
        for items in parsed_event.values():
            items["relationships"] = []
            for relationship in items["position"]:
                if relationship.startswith("interact with"):
                    anchor = relationship.split(" ")[-1]
                    relationship = f"adjacent to {anchor}"
                items["relationships"].append(relationship)
            for relationship in items.get("orientation", []):
                items["relationships"].append(relationship)

            del items["position"]
            if "orientation" in items:
                del items["orientation"]

        # make sure that the target actions include "sit" or "lie"
        # if the relationship includes "sit on" or "lie on"
        for items in parsed_event.values():
            for relationship in items["relationships"]:
                if relationship.startswith("sit on"):
                    items["target_action"].append("sit")
                    break
                elif relationship.startswith("lie on"):
                    items["target_action"].append("lie")
                    break

        # remove duplicate target actions
        for items in parsed_event.values():
            items["target_action"] = list(set(items["target_action"]))

        # store parsed event in the event object
        event.parsed_event = deepcopy(parsed_event)

        return parsed_event

    def _check_validity(self, event: Event, response, available_action_labels):
        if not self.use_self_feedback:
            return ""

        pattern = r"^(" + "|".join(_VALID_RELATIONSHIPS) + ")"

        for character in event.involved_characters:
            if character.name not in response:
                feedback = f'You did not generated output for this character: "{character.name}"'
                return feedback

        for character in response:
            relationships = response[character]["relationships"]
            for relationship in relationships:
                if not re.match(pattern, relationship):
                    feedback = f'You have generated a relationship that is not valid: "{relationship}"'
                    feedback += "\nPlease regenerate a valid relationship specified in the system message."
                    return feedback

                anchor = relationship.split(" ")[-1]
                if anchor not in self.character_names and anchor not in self.object_labels:
                    feedback = f'You have generated a relationship with non-existent entity as an anchor: "{relationship}" (anchor: "{anchor}")'
                    feedback += "\nPlease regenerate a valid relationship using an existing entity in the scene."
                    return feedback

        for character in response:
            target_actions = response[character]["target_action"]
            for target_action in target_actions:
                if target_action not in available_action_labels + [
                    "sit",
                    "lie",
                ]:  # sit and lie are added automatically based on the relationships
                    feedback = f'You have generated an action that is not valid: "{target_action}"'
                    feedback += "\nYou can use only the actions in the available action labels."
                    return feedback
        return ""

    def _get_main_action_label(self, action_labels):
        if not action_labels:
            return ""
        if len(action_labels) == 1:
            return action_labels[0]
        return next(action for action in action_labels if not action.startswith("sit") and not action.startswith("lie"))
