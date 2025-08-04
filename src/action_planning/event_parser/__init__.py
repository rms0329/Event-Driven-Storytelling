from .event_parser import EventParser
from .event_parser_vlm import EventParserVLM
from .event_parser_wo_lcps import EventParserWithoutLcps


def get_event_parser(characters, objects, cfg):
    if cfg.event_parser.disable_lcps:
        return EventParserWithoutLcps(characters, objects, cfg)
    if cfg.scene_describer.description_type == "scene_image":
        return EventParserVLM(characters, objects, cfg)
    return EventParser(characters, objects, cfg)
