from datetime import datetime
from typing import List

from src.action_planning.narrator.narrator import Narrator
from src.type import Character, Object
from src.utils.misc import encode_image


class NarratorVLM(Narrator):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.system_messages_dir = self.base_dir / "system_messages"
        self.system_message_file = self.system_messages_dir / "narrator_vlm.txt"

    def _create_query_message(
        self,
        scene_description,
        available_activities,
        user_instruction=None,
        characters=None,
        previous_events=None,
        feedback=None,
    ):
        def _create_user_content(type: str, data: str):
            if type == "text":
                return {"type": type, type: data}
            elif type == "image_url":
                return {"type": type, type: {"url": f"data:image/jpeg;base64,{data}"}}
            else:
                return ValueError(f"There is no type '{type}', type should be 'text' or 'image_url'")

        # text part
        text_part = super()._create_query_message(
            scene_description[0],
            available_activities,
            user_instruction,
            characters,
            previous_events,
            feedback,
        )
        text_part = text_part.replace("[Scene Description]", "[Object List]")
        text_part = _create_user_content("text", text_part)

        # image part
        scene_image_path = scene_description[1]
        image_data_str = encode_image(scene_image_path)
        image_part = _create_user_content("image_url", image_data_str)

        query_message = [text_part, image_part]
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
                    f.write(message["content"][0]["text"])
                    f.write("\n\n")
                elif i % 2 == 1:
                    f.write(f"{'='*30} Example {(i-1)//2} Query {'='*30}\n")
                    f.write(message["content"][0]["text"])
                    f.write("\n\n")
                else:
                    f.write(f"{'='*30} Example {(i-1)//2} Response {'='*30}\n")
                    f.write(message["content"])
                    f.write("\n\n")

            f.write(f"{'='*30} Response {'='*30}\n")
            f.write(response)
