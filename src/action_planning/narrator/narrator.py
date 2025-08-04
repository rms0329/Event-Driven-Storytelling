import json
import re
from datetime import datetime
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

from src.action_planning.llm_client import LLMClient
from src.motion_synthesis.states import State
from src.type import Character, Event, Object
from src.utils import misc
from src.utils.misc import CustomEncoder


class Narrator:
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        self.client = LLMClient(
            provider=cfg.narrator.provider,
            model=cfg.narrator.model,
        )
        self.characters = characters
        self.objects = objects
        self.cfg = cfg

        # for ablation study / evaluation
        self.disable_event_based_planning = cfg.narrator.disable_event_based_planning
        self.event_based_planning = not self.disable_event_based_planning

        self.model = cfg.narrator.model
        self.temperature = cfg.narrator.temperature
        self.use_json_mode = cfg.narrator.use_JSON_mode
        self.use_examples = cfg.narrator.use_examples
        self.use_self_feedback = cfg.narrator.use_self_feedback
        self.base_dir = Path(__file__).parent
        self.system_messages_dir = self.base_dir / "system_messages"
        self.system_message_file = self.system_messages_dir / "narrator.txt"
        self.additional_guidelines = cfg.narrator.additional_guidelines
        self.examples_dir = self.base_dir / "examples"
        self.example_files = sorted(self.examples_dir.glob("*.yaml"))

        self.logger = misc.get_console_logger(self.__class__.__name__, level=cfg.logging_level)
        self.log_basedir = Path("./logs/narrator")
        self.log_id_prefix = None
        self.scene_descriptions_dir = Path("./src/action_planning/scene_describer/sample_descriptions")
        self.scene_description_type = cfg.scene_describer.description_type

        self.max_self_feedback_count = cfg.narrator.max_self_feedback_count
        self.self_feedback_count = 0
        self.left_query_count = 0
        self.available_characters: List[Character] = []
        self.previous_events: List[Event] = []
        self.skipped_characters: List[Character] = []
        self.finished = False

    def is_new_event_required(self):
        if self.finished or self.left_query_count <= 0:
            return False

        self.available_characters = []
        for character in self.characters:
            if character.state in (
                State.APPROACHING,
                State.TRANSITION_IN,
                State.TRANSITION_OUT,
            ):
                continue

            if character.event is None:
                self.available_characters.append(character)
            elif character.event.state == "looping":
                self.available_characters.append(character)

        if not self.available_characters:
            return False
        if all(c in self.skipped_characters for c in self.available_characters):
            return False
        return True

    def find_available_characters(self, characters):
        available_characters = []
        for character in characters:
            if character.state in (
                "approaching",
                "transition_in",
                "transition_out",
            ):
                continue

            if character.state == "idle":
                if character.event is None:  # truely idle
                    available_characters.append(character)
                else:  # waiting human-human interaction
                    continue
            elif character.state == "interacting":
                if character.event is not None and character.event.state == "looping":
                    available_characters.append(character)
                else:
                    continue
            else:
                raise ValueError(f"Invalid character state: {character.state}")
        return available_characters

    def generate_subsequent_plan(
        self,
        scene_description,
        available_activities,
        user_instruction=None,
        feedback=None,
        with_usage=False,
        accumulated_prompt_tokens=0,
        accumulated_completion_tokens=0,
    ):
        self.logger.info("Generating subsequent plan...")
        response, prompt_tokens, completion_tokens = self._query_llm(
            scene_description,
            available_activities,
            user_instruction,
            feedback,
        )
        self.left_query_count -= 1
        self.left_query_count = max(0, self.left_query_count)

        response = self._postprocess_before_validation(response)
        feedback = self._check_validity(response)
        if feedback and self.self_feedback_count < self.max_self_feedback_count:
            self.logger.warning("Invalid response detected, retrying with feedback...")
            self.logger.warning(f"feedback: {feedback}")
            self.self_feedback_count += 1
            return self.generate_subsequent_plan(
                scene_description,
                available_activities,
                user_instruction,
                feedback,
                with_usage,
                accumulated_prompt_tokens + prompt_tokens,
                accumulated_completion_tokens + completion_tokens,
            )

        response = self._postprocess_after_validation(response)
        if with_usage:
            return (
                response,
                prompt_tokens + accumulated_prompt_tokens,
                completion_tokens + accumulated_completion_tokens,
            )
        return response

    def _query_llm(
        self,
        scene_description,
        available_activities,
        user_instruction=None,
        feedback=None,
    ):
        args, messages = self.create_api_request(
            scene_description,
            available_activities,
            user_instruction,
            feedback,
            return_with_messages=True,
        )
        response, prompt_tokens, completion_tokens = self.client.get_response(with_usage=True, **args)
        self.log(messages, response)
        return response, prompt_tokens, completion_tokens

    def create_api_request(
        self,
        scene_description,
        available_activities,
        user_instruction=None,
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
                    scene_description=scene_description,
                    available_activities=available_activities,
                    user_instruction=user_instruction,
                    feedback=feedback,
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
        example_data = OmegaConf.load(example_file)
        example_data = OmegaConf.to_container(example_data, resolve=True)

        # create user message part
        scene_description = (
            (self.scene_descriptions_dir / f"{example_data['scene_name']}_{self.scene_description_type}.txt")
            .read_text()
            .strip()
        )
        if self.scene_description_type == "scene_image":
            img_path = f"./configs/scenes/{example_data['scene_name']}/top_view.png"
            scene_description = (scene_description, img_path)
        available_activities = example_data["action_labels"]
        user_instruction = example_data["user_instruction"]

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
                    state=event["state"],
                )
            )

            # assign event to characters
            for c in characters:
                if c.name not in event["characters"]:
                    continue
                if event["state"] == "completed":
                    c.event = None
                else:
                    c.event = previous_events[-1]

            # assign parsed data to event
            parsed_event = {}
            for parsed in event["parsed"]:
                parsed_event[parsed["character"]] = {
                    "target_action": [parsed["target_action"]],
                    "relationships": parsed["position"] + parsed["orientation"],
                }
            previous_events[-1].parsed_event = parsed_event

        user_part = self._create_query_message(
            scene_description,
            available_activities,
            user_instruction,
            characters,
            previous_events,
        )

        # create assistant message part
        reasoning = example_data["response"]["reasoning"].replace("\n\n", "\n")
        reasoning = reasoning.replace("\n", " ").replace("event/plan", "event").strip()
        assistant_part = {
            "reasoning": reasoning,
            "event": {
                "characters": example_data["response"]["characters"],
                "activity": example_data["response"]["activity"],
            },
        }
        assistant_part = json.dumps(assistant_part, cls=CustomEncoder)
        return user_part, assistant_part

    def _create_query_message(
        self,
        scene_description,
        available_activities,
        user_instruction=None,
        characters=None,
        previous_events=None,
        feedback=None,
    ):
        if characters is None:
            characters = self.characters
        if previous_events is None:
            previous_events = self.previous_events

        scene_description = f"[Scene Description]\n{scene_description}"
        available_activities = f"[Available Activities]\n{available_activities}"
        user_instruction = f"[User Instruction]\n{user_instruction if user_instruction else 'None'}"
        character_information = "[Available Characters]\n"
        for character in self.find_available_characters(characters):
            character_information += f"- {character.name}\n"

        event_history = "[Event History]\n"
        for i, event in enumerate(previous_events):
            event_history += f"- Event {i}: {event} ({event.state})\n"
        if not previous_events:
            event_history += "None"
        feedback = f"[Feedback]\n{feedback if feedback else 'None'}"

        query_message = map(
            str.strip,
            [scene_description, available_activities, user_instruction, character_information, event_history, feedback],
        )
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

    def _postprocess_before_validation(self, response):
        response = json.loads(response)
        if not response.get("event", None):
            return response

        # post process activity part
        activity = response["event"]["activity"]
        activity = activity.replace("[", "").replace("]", "")
        activity = re.sub(r"\s\(.*?\)$", "", activity)  # remove event state
        response["event"]["activity"] = activity

        return response

    def _check_validity(self, response):
        if not self.use_self_feedback:
            return ""
        if not response.get("event", None):
            return ""

        characters = response["event"]["characters"]
        if not all(c in self.available_characters for c in characters):
            feedback = "You have generated an event with an invalid character. Characters in the event should be from the available characters."
            feedback += f"\n  - Available characters: {[c.name for c in self.available_characters]}"
            feedback += f"\n  - Characters in the event: {characters}"
            return feedback

        return ""

    def _postprocess_after_validation(self, response):
        # when LLM wants to skip the turn
        if not response.get("event", None) or not response["event"]["characters"]:
            self.skipped_characters = self.available_characters.copy()
            return None

        characters = response["event"]["characters"]
        activity = response["event"]["activity"]
        event = Event(
            involved_characters=[c for c in self.characters if c.name in characters],
            activity=activity,
        )
        event.reasoning = response.get("reasoning", None)
        if not event.involved_characters:
            raise RuntimeError(
                f"No matched characters found in the response: {characters}. Available characters: {self.characters}"
            )
        self.previous_events.append(event)
        self.skipped_characters = []
        self.self_feedback_count = 0
        return event

    def _get_main_action_label(self, action_labels):
        if not action_labels:
            return ""
        if len(action_labels) == 1:
            return action_labels[0]
        return next(action for action in action_labels if not action.startswith("sit") and not action.startswith("lie"))
