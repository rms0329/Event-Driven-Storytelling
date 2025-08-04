import torch

from src.type import Character, Role
from src.utils import misc, smpl
from src.utils.transform import quaternion_multiply, quaternion_relative_rotation, quaternion_to_axis_angle

from ..states import State
from ..utils import feature as mm
from ..utils import spring
from .motion_db import MotionDB


class MotionMatcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.motion_db = MotionDB(cfg)
        self.device = cfg.device
        self.target_framerate = cfg.framerate
        self.dt = 1 / self.target_framerate
        self._motion_update_frequency = cfg.motion_matching.motion_update_frequency
        self.min_grid_path_len = cfg.path_planner.min_planning_timesteps + 1
        self.halflife_transl = cfg.motion_matching.halflife_transl
        self.halflife_global_orient = cfg.motion_matching.halflife_global_orient
        self.halflife_pose = cfg.motion_matching.halflife_pose
        self.halflife_walk = cfg.motion_matching.halflife_walk
        self.halflife_hsi = cfg.motion_matching.halflife_hsi
        self.halflife_hhi = cfg.motion_matching.halflife_hhi
        self.halflife_idle = cfg.motion_matching.halflife_idle
        self.hhi_tags = self.motion_db.hhi_tags
        self.hsi_tags = self.motion_db.hsi_tags
        self.hsi_matching_threshold = cfg.state_manager.hsi_matching_threshold

    @property
    def motion_update_frequency(self):
        return self._motion_update_frequency

    @motion_update_frequency.setter
    def motion_update_frequency(self, value):
        self._motion_update_frequency = value
        self.motion_db.motion_update_frequency = value

    def update_best_matching_frame(self, character: Character):
        if character._use_partwise_mm:
            return self._update_best_matching_frame_partwise(character)

        tag = character.main_action
        new_fId = self.motion_db.search_best_matching_frame(character, tag)
        matching_feature = self.motion_db.get_matching_feature(new_fId, tag)

        if new_fId != character.fId:
            # inertialization offset should be reset
            character.invoke_inertialization = True

        character.fId = new_fId
        character.num_played_frames = 0
        character.matching_feature = matching_feature

    def advance_motion_frame(self, character: Character):
        if character._use_partwise_mm:
            return self._advance_motion_frame_partwise(character)

        if character.play_synchronized_motion:
            character.fId = character.interactee.fId
            next_pose_vector = self.motion_db.get_synchronized_pose_feature(
                character.interactee.fId,
                character.interactee.main_action,
            )
        else:
            next_pose_vector = self.motion_db.get_pose_feature(
                character.fId,
                character.main_action,
            )
        next_body_params = mm.advance_frame(character.body_params, next_pose_vector, self.dt)

        # if the character is in sitting posture, we do not use the z translation from the motion clip
        # instead, we use the z translation from the previous frame and rely on the adjustment
        # this is to enable the character to sit on the tall chairs that are not included in the motion database
        if (character.state == State.INTERACTING or character.state == State.IDLE) and character.is_resting_posture():
            next_body_params[:, 2:3] = character.body_params[-1:, 2:3]

        # if motion clip is changed, inertialization offset should be reset
        # based on the current frame and the next target frame
        if character.invoke_inertialization:
            next_root_velocity = next_pose_vector[9:12].unsqueeze(0)
            next_root_angular_velocity = next_pose_vector[16:19].unsqueeze(0)
            next_joint_angular_velocities = next_pose_vector[103:].unsqueeze(0)
            self.reset_inertialization_offset(
                character,
                next_body_params,
                next_root_velocity,
                next_root_angular_velocity,
                next_joint_angular_velocities,
            )

        # inertialization offset decays every frame
        self.decay_inertialization_offset(character)

        # apply offset to next target frame
        next_body_params[:, :3] += character.offset_x[:, :3]
        next_body_params[:, 3:7] = quaternion_multiply(character.offset_x[:, 3:7], next_body_params[:, 3:7])
        next_body_params[:, 7:] = quaternion_multiply(
            character.offset_x[:, 7:].reshape(1, 21, 4),
            next_body_params[:, 7:].reshape(1, 21, 4),
        ).reshape(1, 84)

        # adjust character's position based on its state
        next_body_params = self.adjust_character_position(character, next_body_params)

        # append the next body parameters to the character's body parameters
        character.body_params = torch.cat([character.body_params, next_body_params], dim=0)
        character.fId += 1
        character.num_played_frames += 1

    def adjust_character_position(self, character: Character, next_body_params):
        # when the character is approaching state,
        # adjust character's position close to its planned path
        if character.state == State.APPROACHING:
            next_position = smpl.get_character_positions(next_body_params)
            distances = torch.norm(character.spline_curve - next_position, dim=1, keepdim=True)
            closest_idx = torch.argmin(distances)
            target_position = character.spline_curve[closest_idx].reshape(1, 3)

            adjustment = spring.damp_adjustment_exact_vector(
                target_position - next_position,
                self.halflife_walk,
                self.dt,
            )
            next_body_params[:, :3] += adjustment

        # when the character is in sitting state,
        # adjust character's position and rotation to its target position and rotation
        elif (character.state == State.INTERACTING or character.state == State.IDLE) and character.is_resting_posture():
            next_root_position = smpl.get_root_positions(next_body_params)
            next_facing_direction = smpl.get_facing_directions(next_body_params)
            next_rotation = smpl.get_character_rotations(next_facing_direction)
            target_root_position = character.target_root_position
            target_facing_direction = character.target_facing_direction
            target_rotation = smpl.get_character_rotations(target_facing_direction)

            position_adjustment = spring.damp_adjustment_exact_vector(
                target_root_position - next_root_position,
                self.halflife_hsi,
                self.dt,
            )
            rotation_adjustment = spring.damp_adjustment_exact_quaternion(
                quaternion_relative_rotation(
                    rot_from=next_rotation,
                    rot_to=target_rotation,
                ),
                self.halflife_hsi,
                self.dt,
            )
            next_body_params[:, :3] += position_adjustment
            next_body_params[:, 3:7] = quaternion_multiply(rotation_adjustment, next_body_params[:, 3:7])

        # when the character is in transition_in state,
        # we adjust the character's position to ensure the successful transition
        elif character.state == State.TRANSITION_IN:
            position_adjustment = spring.damp_adjustment_exact_vector(
                character.target_root_position - character.matched_root_position,
                self.halflife_hsi,
                self.dt,
            )
            if character.accumulated_adjustment is None:
                character.accumulated_adjustment = position_adjustment
            else:
                character.accumulated_adjustment += position_adjustment
            next_body_params[:, :2] += position_adjustment[:, :2]
            next_body_params[:, 2:3] += character.accumulated_adjustment[:, 2:3]  # since z comes from the motion clip
            character.matched_root_position += position_adjustment

        # when the character is interacting with another character,
        # adjust character's position and rotation based on the original motion clip
        # (for easy implementation, only the reactor character is adjusted)
        elif (
            character.state == State.INTERACTING
            and character.interactee is not None
            and character.role == Role.REACTOR
            and not character._use_partwise_mm
        ):
            next_position = smpl.get_character_positions(next_body_params)
            next_facing_direction = smpl.get_facing_directions(next_body_params)
            next_rotation = smpl.get_character_rotations(next_facing_direction)
            (
                target_position,
                target_rotation,
            ) = self.get_desired_position_and_rotation(character)

            position_adjustment = spring.damp_adjustment_exact_vector(
                target_position - next_position,
                self.halflife_hhi,
                self.dt,
            )
            rotation_adjustment = spring.damp_adjustment_exact_quaternion(
                quaternion_relative_rotation(
                    rot_from=next_rotation,
                    rot_to=target_rotation,
                ),
                self.halflife_hhi,
                self.dt,
            )
            next_body_params[:, :3] += position_adjustment
            next_body_params[:, 3:7] = quaternion_multiply(rotation_adjustment, next_body_params[:, 3:7])

        elif (
            character.state == State.IDLE
            and not character.current_actions
            and not character.target_actions
            and character.is_stationary()
        ):
            next_position = smpl.get_character_positions(next_body_params)
            next_facing_direction = smpl.get_facing_directions(next_body_params)
            next_rotation = smpl.get_character_rotations(next_facing_direction)
            target_position = character.target_position
            target_facing_direction = character.target_facing_direction
            target_rotation = smpl.get_character_rotations(target_facing_direction)

            position_adjustment = spring.damp_adjustment_exact_vector(
                target_position - next_position,
                self.halflife_idle,
                self.dt,
            )
            rotation_adjustment = spring.damp_adjustment_exact_quaternion(
                quaternion_relative_rotation(
                    rot_from=next_rotation,
                    rot_to=target_rotation,
                ),
                self.halflife_idle,
                self.dt,
            )
            next_body_params[:, :3] += position_adjustment
            next_body_params[:, 3:7] = quaternion_multiply(rotation_adjustment, next_body_params[:, 3:7])

        # adjust character's rotation based on the target facing direction
        # (when 'look at' relationship is established)
        elif (
            character.state == State.INTERACTING
            and character.interactee is None
            and character.target_facing_direction is not None
            and not character.is_resting_posture()
        ):
            next_facing_direction = smpl.get_facing_directions(next_body_params)
            next_rotation = smpl.get_character_rotations(next_facing_direction)
            target_facing_direction = character.target_facing_direction
            target_rotation = smpl.get_character_rotations(target_facing_direction)
            rotation_adjustment = spring.damp_adjustment_exact_quaternion(
                quaternion_relative_rotation(
                    rot_from=next_rotation,
                    rot_to=target_rotation,
                ),
                self.halflife_hsi,
                self.dt,
            )
            next_body_params[:, 3:7] = quaternion_multiply(rotation_adjustment, next_body_params[:, 3:7])

        return next_body_params

    def reset_inertialization_offset(
        self,
        character: Character,
        target_body_params,
        next_root_velocity,
        next_root_angular_velocity,
        next_joint_angular_velocities,
    ):
        curr_transl = character.body_params[-1:, :3]
        prev_transl = character.body_params[-2:-1, :3]
        next_transl = target_body_params[:, :3]
        curr_root_rotation = character.body_params[-1:, 3:7]
        prev_root_rotation = character.body_params[-2:-1, 3:7]
        next_root_rotation = target_body_params[:, 3:7]
        curr_joint_rotations = character.body_params[-1:, 7:].reshape(1, 21, 4)
        prev_joint_rotations = character.body_params[-2:-1, 7:].reshape(1, 21, 4)
        next_joint_rotations = target_body_params[:, 7:].reshape(1, 21, 4)

        # calculate the offset between the current frame and the next target frame
        root_position_offset = curr_transl - next_transl
        root_velocity_offset = ((curr_transl - prev_transl) / self.dt) - next_root_velocity

        root_rotation_offset = quaternion_relative_rotation(next_root_rotation, curr_root_rotation)
        root_angular_velocity_offset = (
            quaternion_to_axis_angle(quaternion_relative_rotation(prev_root_rotation, curr_root_rotation)) / self.dt
            - next_root_angular_velocity
        )

        joint_rotation_offset = quaternion_relative_rotation(next_joint_rotations, curr_joint_rotations).reshape(1, 84)
        joint_angular_velocity_offset = (
            quaternion_to_axis_angle(quaternion_relative_rotation(prev_joint_rotations, curr_joint_rotations)) / self.dt
        ).reshape(1, 63) - next_joint_angular_velocities

        # set the offset to the character's offset variable
        character.offset_x = torch.cat([root_position_offset, root_rotation_offset, joint_rotation_offset], dim=1)
        character.offset_v = torch.cat(
            [
                root_velocity_offset,
                root_angular_velocity_offset,
                joint_angular_velocity_offset,
            ],
            dim=1,
        )
        character.invoke_inertialization = False

    def decay_inertialization_offset(self, character: Character):
        transl_x, transl_v = spring.decay_spring_damper_vector(
            character.offset_x[:, :3],
            character.offset_v[:, :3],
            self.halflife_transl,
            self.dt,
        )
        global_orient_x, global_orient_v = spring.decay_spring_damper_quaternion(
            character.offset_x[:, 3:7],
            character.offset_v[:, 3:6],
            self.halflife_global_orient,
            self.dt,
        )
        pose_x, pose_v = spring.decay_spring_damper_quaternion(
            character.offset_x[:, 7:].reshape(1, 21, 4),
            character.offset_v[:, 6:].reshape(1, 21, 3),
            self.halflife_pose,
            self.dt,
        )
        character.offset_x = torch.cat([transl_x, global_orient_x, pose_x.reshape(1, 84)], dim=1)
        character.offset_v = torch.cat([transl_v, global_orient_v, pose_v.reshape(1, 63)], dim=1)

    def is_motion_update_needed(self, character: Character):
        if character._use_partwise_mm:
            return self._is_motion_update_needed_partwise(character)

        if character.enforce_motion_search:
            character.enforce_motion_search = False
            return True

        # if the animation reaches the end, update is needed
        if self.motion_db.is_end_of_motion(character.fId, character.main_action):
            return True

        # in transition states, we just want to play the motion without interruption
        if character.state in (State.TRANSITION_IN, State.TRANSITION_OUT):
            return False

        # if the character has played enough frames, update is needed
        # however, if the character is interacting, we don't want to interrupt the motion
        if character.num_played_frames >= self.motion_update_frequency:
            return character.state != State.INTERACTING
        return False

    def get_matching_feature(self, character: Character):
        return self.motion_db.get_matching_feature(character.fId, character.current_actions[0])

    def get_desired_position_and_rotation(self, character: Character):
        assert character.interactee is not None and character.role == Role.REACTOR
        assert not character._use_partwise_mm, "Used only when full-body motion matching"
        interactee = character.interactee
        interactee_position = smpl.get_character_positions(interactee.body_params[-1:])
        interactee_facing_direction = smpl.get_facing_directions(interactee.body_params[-1:])  # fmt: skip
        interactee_rotation = smpl.get_character_rotations(interactee_facing_direction)

        tag = character.interactee.main_action
        assert tag in self.hhi_tags, f"This function is supposed to be used for HHI scenarios only, but got {tag}"  # fmt: skip
        desired_position = self.motion_db.get_matching_feature(character.interactee.fId, tag, "relative_position")
        desired_position = misc.add_z(desired_position).reshape(1, 3)
        desired_position = mm.to_world(
            desired_position,
            interactee_position,
            interactee_rotation,
        )
        desired_facing_direction = self.motion_db.get_matching_feature(
            character.interactee.fId, tag, "relative_direction"
        )
        desired_facing_direction = misc.add_z(desired_facing_direction).reshape(1, 3)
        desired_rotation = smpl.get_character_rotations(desired_facing_direction)
        desired_rotation = mm.to_world(desired_rotation, interactee_rotation)
        return desired_position, desired_rotation

    def _is_motion_update_needed_partwise(self, character: Character):
        if character.enforce_motion_search:
            character.enforce_motion_search = False
            character._update_required = [True, True]
            return True

        # in transition states, we just want to play the motion without interruption
        if character.state in (State.TRANSITION_IN, State.TRANSITION_OUT):
            character._update_required = [False, False]
            return False

        # if any of the body parts reaches the end of the motion clip, update is needed
        character._update_required = [
            self.motion_db.is_end_of_motion(character._fIds[0], character.upper_body_action),  # fmt: skip
            self.motion_db.is_end_of_motion(character._fIds[1], character.lower_body_action),  # fmt: skip
        ]
        if any(character._update_required):
            return True

        # if the character has played enough frames, update is needed
        # however, if the character is interacting, we don't want to interrupt the upper body motion
        if character.num_played_frames >= self.motion_update_frequency:
            character._update_required = [character.state != State.INTERACTING, True]
            return True

        return False

    def _update_best_matching_frame_partwise(self, character: Character):
        new_fIds = character._fIds.copy()
        if character._update_required[0]:
            new_fIds[0] = self.motion_db.search_best_matching_frame(character, character.upper_body_action)
        if character._update_required[1]:
            new_fIds[1] = self.motion_db.search_best_matching_frame(character, character.lower_body_action)

        if new_fIds != character._fIds:
            # inertialization offset should be reset
            character.invoke_inertialization = True

        character._fIds = new_fIds
        character.num_played_frames = 0

    def _advance_motion_frame_partwise(self, character: Character):
        if character.play_synchronized_motion:  # this only applies to the upper body
            character._fIds[0] = character.interactee._fIds[0]
            next_pose_vector_upper = self.motion_db.get_synchronized_pose_feature(character._fIds[0], character.upper_body_action)  # fmt: skip
            next_pose_vector_lower = self.motion_db.get_pose_feature(character._fIds[1], character.lower_body_action)  # fmt: skip
        else:
            next_pose_vector_upper = self.motion_db.get_pose_feature(character._fIds[0], character.upper_body_action)  # fmt: skip
            next_pose_vector_lower = self.motion_db.get_pose_feature(character._fIds[1], character.lower_body_action)  # fmt: skip

        next_pose_vector = mm.replace_upper_body(
            src_pose_feature=next_pose_vector_lower,
            rpl_pose_feature=next_pose_vector_upper,
        )
        next_body_params = mm.advance_frame(character.body_params, next_pose_vector, self.dt)

        # if the character is in sitting posture, we do not use the z translation from the motion clip
        # instead, we use the z translation from the previous frame and rely on the adjustment
        # this is to enable the character to sit on the tall chairs that are not included in the motion database
        if (character.state == State.INTERACTING or character.state == State.IDLE) and character.is_resting_posture():
            next_body_params[:, 2:3] = character.body_params[-1:, 2:3]

        # if motion clip is changed, inertialization offset should be reset
        # based on the current frame and the next target frame
        if character.invoke_inertialization:
            next_root_velocity = next_pose_vector[9:12].unsqueeze(0)
            next_root_angular_velocity = next_pose_vector[16:19].unsqueeze(0)
            next_joint_angular_velocities = next_pose_vector[103:].unsqueeze(0)
            self.reset_inertialization_offset(
                character,
                next_body_params,
                next_root_velocity,
                next_root_angular_velocity,
                next_joint_angular_velocities,
            )

        # inertialization offset decays every frame
        self.decay_inertialization_offset(character)

        # apply offset to next target frame
        next_body_params[:, :3] += character.offset_x[:, :3]
        next_body_params[:, 3:7] = quaternion_multiply(character.offset_x[:, 3:7], next_body_params[:, 3:7])
        next_body_params[:, 7:] = quaternion_multiply(
            character.offset_x[:, 7:].reshape(1, 21, 4),
            next_body_params[:, 7:].reshape(1, 21, 4),
        ).reshape(1, 84)

        # adjust character's position based on its state
        next_body_params = self.adjust_character_position(character, next_body_params)

        # append the next body parameters to the character's body parameters
        character.body_params = torch.cat([character.body_params, next_body_params], dim=0)
        character._fIds[0] += 1
        character._fIds[1] += 1
        character.num_played_frames += 1
