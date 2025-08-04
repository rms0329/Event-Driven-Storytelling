from copy import deepcopy
from pathlib import Path
from typing import List

import yaml

from src.action_planning.event_parser.code_executor import CodeExecutor
from src.action_planning.narrator.narrator import Narrator
from src.type import Character, Event, Object


class NarratorWithoutEvent(Narrator):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.use_json_mode = False  # we cannot use json mode when using Pythonic prompting
        self.system_message_file = self.system_messages_dir / "narrator_wo_event.txt"
        self.code_executor = CodeExecutor(characters, objects, cfg)
        assert self.disable_event_based_planning

    def _create_example_message(self, example_file: Path):
        assert example_file.suffix == ".yaml"

        # create user message part
        user_part, _ = super()._create_example_message(example_file)

        # create assistant message part
        example_data = yaml.safe_load(example_file.read_text())
        assistant_part = example_data["response_wo_event"].strip()

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

        planning_history = "[Planning History]\n"
        for i, event in enumerate(previous_events):
            planning_history += f"- Plan {i}:\n"
            parsed_event = event.parsed_event
            for character_name, plan in parsed_event.items():
                planning_history += f"  - {character_name}:\n"
                planning_history += f"    - target_action: '{self._get_main_action_label(plan['target_action'])}'\n"
                planning_history += f"    - relationships: {plan['relationships']}\n"
                planning_history += f"    - state: '{event.state}'\n"

        if not previous_events:
            planning_history += "None"
        feedback = f"[Feedback]\n{feedback if feedback else 'None'}"

        query_message = map(
            str.strip,
            [
                scene_description,
                available_activities,
                user_instruction,
                character_information,
                planning_history,
                feedback,
            ],
        )
        query_message = "\n\n".join(query_message).strip()
        return query_message

    def _postprocess_before_validation(self, response):
        self.code_executor.set_previous_events(self.previous_events)
        response = self.code_executor.execute_response(response)
        if not response:
            return {}

        # convert target_action to a list
        for character_name, plan in response.items():
            response[character_name]["target_action"] = [plan["target_action"]]

        # integrate relationships
        for character_name, plan in response.items():
            response[character_name]["relationships"] = []
            for relationship in plan["position"] + plan.get("orientation", []):
                if relationship.startswith("interact with"):
                    anchor = relationship.split(" ")[-1]
                    relationship = f"adjacent to {anchor}"
                response[character_name]["relationships"].append(relationship)

            del response[character_name]["position"]
            if "orientation" in response[character_name]:
                del response[character_name]["orientation"]

        # make sure that the target actions include "sit" or "lie"
        # if the relationship includes "sit on" or "lie on"
        for character_name, plan in response.items():
            for relationship in plan["relationships"]:
                if relationship.startswith("sit on"):
                    response[character_name]["target_action"].append("sit")
                    break
                elif relationship.startswith("lie on"):
                    response[character_name]["target_action"].append("lie")
                    break

        # remove duplicate target actions
        for character_name, plan in response.items():
            response[character_name]["target_action"] = list(set(plan["target_action"]))
        return response

    def _check_validity(self, response):
        if not self.use_self_feedback:
            return ""

        characters = list(response.keys())
        if not all(c in self.available_characters for c in characters):
            feedback = "You have generated an event with an invalid character. Characters in the event should be from the available characters."
            feedback += f"\n  - Available characters: {[c.name for c in self.available_characters]}"
            feedback += f"\n  - Characters in the event: {characters}"
            return feedback

        return ""

    def _postprocess_after_validation(self, response):
        characters = [c for c in self.characters if c.name in response.keys()]
        if not characters:  # when LLM wants to skip the turn
            self.skipped_characters = self.available_characters.copy()
            return None

        event = Event(
            involved_characters=characters,
            activity="dummy_activity",
        )
        event.parsed_event = deepcopy(response)
        if not event.involved_characters:
            raise RuntimeError(
                f"No matched characters found in the response: {characters}. Available characters: {self.characters}"
            )
        self.previous_events.append(event)
        self.skipped_characters = []
        self.self_feedback_count = 0
        return response  # already in parsed event format
