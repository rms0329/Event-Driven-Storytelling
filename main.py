import threading
import time

import numpy as np

from src.action_planning.action_planning_module import ActionPlanningModule
from src.motion_synthesis.motion_synthesis_module import MotionSynthesisModule
from src.scene.scene import Scene
from src.type import Character
from src.utils import misc
from src.utils.polyscope import PolyscopeApp

_DEFAULT_CHARACTER_NAMES = [
    "Amir",
    "Benjamin",
    "Charles",
    "David",
    "Evan",
    "Francis",
]
_DEFAULT_COLOR = [0.7, 0.7, 0.7]
_DEFAULT_STORY = """
Amir makes coffee in the kitchen.
On the other hand, Benjamin works with his laptop sitting on the sofa.
After making coffee, Amir sits in a near chair to Benjamin and drinks coffee.
Charles and David are having a conversation in the dining area.
"""


class DemoApp(PolyscopeApp):
    def __init__(self, cfg):
        super().__init__(framerate=cfg.framerate)
        self.cfg = cfg
        self.device = cfg.device
        self.scene_name = cfg.scene_name
        self.num_characters = cfg.num_characters
        self.character_names = _DEFAULT_CHARACTER_NAMES[: self.num_characters]
        self.user_instruction = _DEFAULT_STORY.strip()
        self.playing = False
        self.error_occurred = False
        self.is_status_ui_active = True
        self.use_non_blocking_planning = True
        self.logger = misc.get_console_logger("DemoApp", cfg.logging_level)

        threading.Thread(
            target=self.action_planning_daemon,
            daemon=True,
        ).start()
        self.reset()

    def action_planning_daemon(self):
        while True:
            time.sleep(0.1)
            if not self.use_non_blocking_planning:
                continue
            if not self.playing or self.error_occurred:
                continue
            if self.fId <= self.max_fId:
                continue

            if self.action_planner.is_new_event_required():
                try:
                    self.action_planner.generate_new_event(self.user_instruction)
                except Exception as e:
                    self.logger.error(f"Error occurred while generating new event: {e}")
                    self.error_occurred = True
                    self.error(f"Error occurred while generating new event: {e}")
            time.sleep(0.3)

    def reset(self):
        super().reset()
        self.scene = Scene(self.scene_name, self.cfg)
        self.register_trimesh("scene", self.scene.scene_mesh)

        self.characters = [Character(name) for name in self.character_names]
        self.character_meshes = []
        self.motion_synthesizer = MotionSynthesisModule(self.scene, self.characters, self.cfg)
        self.action_planner = ActionPlanningModule(
            self.scene, self.characters, self.motion_synthesizer.state_manager, self.cfg
        )
        for character in self.characters:
            character.state = self.motion_synthesizer.state_manager.idle_state
            character.body_params = self.motion_synthesizer.create_initial_body_params(
                position=self.scene.grid_map.get_random_position(),
                facing_direction=np.random.rand(2),
            )
            character.current_actions = []
            character.target_position = character.position
            character.target_root_position = character.position
            character.target_facing_direction = character.facing_direction
            character.future_positions = character.position.tile(3, 1)
            character.future_facing_directions = character.facing_direction.tile(3, 1)

            vertices, faces = self.motion_synthesizer.get_current_mesh(character)
            mesh = self.register_mesh(character.name, vertices, faces, color=_DEFAULT_COLOR)
            self.character_meshes.append(mesh)

        self.max_fId = len(self.characters[0].body_params) - 1
        self.fId = len(self.characters[0].body_params)
        self.motion_synthesizer.state_manager.fId = self.fId

    def callback(self):
        if self.fId <= self.max_fId:  # when showing previous frames
            for character, character_mesh in zip(self.characters, self.character_meshes):
                vertices, _ = self.motion_synthesizer.get_mesh_at(character, self.fId)
                character_mesh.update_vertex_positions(vertices)
            if self.playing:
                self.fId += 1
            return

        if not self.playing:
            return
        if self.error_occurred:
            return

        # if non-blocking planning is disabled, planning is done in the main thread (here)
        # in other case, planning is done in a separate thread (`action_planning_daemon`)
        if not self.use_non_blocking_planning:
            if self.action_planner.is_new_event_required():
                try:
                    self.action_planner.generate_new_event(self.user_instruction)
                except Exception as e:
                    self.logger.error(f"Error occurred while generating new event: {e}")
                    self.error_occurred = True
                    self.error(f"Error occurred while generating new event: {e}")

        # if a new event is created, it's allocated to the involved characters
        if not self.action_planner.event_queue.empty():
            event, parsed_event = self.action_planner.event_queue.get()
            event.created_fId = self.fId
            try:
                self.action_planner.allocate_new_plan(event, parsed_event)
            except Exception as e:
                self.logger.error(f"Error occurred while allocating new plan: {e}")
                self.error_occurred = True
                self.error(f"Error occurred while allocating new plan: {e}")

        # update character state / motion
        for character, character_mesh in zip(self.characters, self.character_meshes):
            self.motion_synthesizer.advance_motion_frame(character)

        # update character mesh
        for character, character_mesh in zip(self.characters, self.character_meshes):
            vertices, _ = self.motion_synthesizer.get_current_mesh(character)
            character_mesh.update_vertex_positions(vertices)

        # update frame idx
        self.fId += 1
        self.max_fId = self.fId - 1
        self.motion_synthesizer.state_manager.fId = self.fId
        assert self.max_fId == len(self.characters[0].body_params) - 1

    def ui_callback(self):
        self.gui.SetNextWindowSize((1000, 200), self.gui.ImGuiCond_FirstUseEver)
        self.gui.Begin("Status", self.is_status_ui_active)
        self.gui.TextUnformatted(f"Characters: {self.character_names}")
        self.gui.Separator()
        if self.gui.TreeNode("Available Action Labels"):
            for action_label in self.motion_synthesizer.motion_matcher.motion_db.tags:
                self.gui.TextUnformatted(f" - {action_label}")
            self.gui.TreePop()
        if self.gui.TreeNode("Events"):
            for event in self.action_planner.narrator.previous_events:
                if self.gui.TreeNode(f"{event} ({event.state})"):
                    if event.parsed_event is not None:
                        for character, parsed in event.parsed_event.items():
                            self.gui.TextUnformatted(f" - {character}:")
                            self.gui.TextUnformatted(f"  - Target Actions: {parsed['target_action']}")
                            self.gui.TextUnformatted(f"  - Relationships: {parsed['relationships']}")
                    self.gui.TreePop()
            self.gui.TreePop()

        if self.gui.TreeNode("Characters"):
            for character in self.characters:
                if self.gui.TreeNode(f"{character.name}"):
                    self.gui.TextUnformatted(
                        f"Event: {character.event} ({character.event.state if character.event is not None else ''})"
                    )
                    self.gui.TextUnformatted(f"State: {character.state}")
                    self.gui.TextUnformatted(f"Current Actions: {character.current_actions}")
                    self.gui.TextUnformatted(f"Target Actions: {character.target_actions}")
                    if self.gui.TreeNode("Relationships: "):
                        for relationship in character.relationships:
                            self.gui.TextUnformatted(f" - {relationship}")
                        self.gui.TreePop()
                    self.gui.TreePop()
            self.gui.TreePop()
        self.gui.End()

        self.gui.Text(f"Error Occurred: {self.error_occurred}")
        self.gui.Text(f"Left Query Count: {self.action_planner.narrator.left_query_count}")
        self.gui.Text(f"Narrator: {self.action_planner.narrator.model} (temperature: {self.action_planner.narrator.temperature})")  # fmt: skip
        self.gui.Text(f"Event Parser: {self.action_planner.event_parser.model} (temperature: {self.action_planner.event_parser.temperature})")  # fmt: skip

        _, self.fId = self.gui.SliderInt("Frame", self.fId, 0, self.max_fId)

        if self.gui.Button(("Play" if not self.playing else "Pause")):
            self.playing = not self.playing

        self.gui.SameLine()
        if self.gui.Button("Reset"):
            self.playing = False
            self.reset()

        self.gui.SameLine()
        if self.gui.Button("Increase Query Budget"):
            self.action_planner.narrator.left_query_count += 1

        _, self.use_non_blocking_planning = self.gui.Checkbox(
            "Use Non-Blocking Planning", self.use_non_blocking_planning
        )
        _, self.user_instruction = self.gui.InputTextMultiline("User Instruction", self.user_instruction)


def main():
    cfg = misc.load_cfg("./configs/demo.yaml")
    app = DemoApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
