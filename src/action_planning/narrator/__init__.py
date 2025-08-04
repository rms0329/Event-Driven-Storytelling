from .narrator import Narrator
from .narrator_vlm import NarratorVLM
from .narrator_wo_event import NarratorWithoutEvent


def get_narrator(characters, objects, cfg):
    if cfg.narrator.disable_event_based_planning:
        return NarratorWithoutEvent(characters, objects, cfg)
    if cfg.scene_describer.description_type == "scene_image":
        return NarratorVLM(characters, objects, cfg)
    return Narrator(characters, objects, cfg)
