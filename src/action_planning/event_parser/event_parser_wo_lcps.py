from copy import deepcopy

import yaml

from src.type import Event

from .event_parser import EventParser


class EventParserWithoutLcps(EventParser):
    def __init__(self, characters, objects, cfg):
        super().__init__(characters, objects, cfg)
        assert self.scene_description_type == "scene_graph", "EventParserWithoutLcps only supports scene_graph"
        self.system_message_file = self.system_messages_dir / "event_parser_wo_lcps.txt"

    def _create_example_message(self, example_file):
        assert example_file.suffix == ".yaml"
        assert example_file.stem == "PI"
        user_part, _ = super()._create_example_message(example_file)

        # create assistant message part
        example_data = yaml.safe_load(example_file.read_text())
        assistant_part = example_data["response_wo_lcps"].strip()
        return user_part, assistant_part

    def _post_process_response(self, event: Event, response):
        parsed_event = self.code_executor.execute_response(response)

        for items in parsed_event.values():
            items["target_action"] = [items["target_action"]]

        for items in parsed_event.values():
            items["relationships"] = []
            items["sampled_position"] = items["position"].copy()
            del items["position"]

        # remove duplicate target actions
        for items in parsed_event.values():
            items["target_action"] = list(set(items["target_action"]))

        # store parsed event in the event object
        event.parsed_event = deepcopy(parsed_event)

        return parsed_event
