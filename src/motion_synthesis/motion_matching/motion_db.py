from functools import lru_cache
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.type import Character
from src.utils import amass, inter_x, samp, smpl

from ..utils import feature as mm


class MotionDB:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.target_framerate = cfg.framerate
        self.dt = 1 / self.target_framerate
        self.motion_update_frequency = cfg.motion_matching.motion_update_frequency
        self.future_offsets = [int(offset * self.target_framerate) for offset in cfg.motion_matching.future_offsets]
        self.num_future_offsets = len(self.future_offsets)
        self.db_save_file = Path("./data/motion_db.pt")
        self.db_config_dir = Path("./configs/motion_db")
        self.tags = sorted([f.stem for f in self.db_config_dir.glob("*.yaml")])

        self.matching_db = {}
        self.animation_db = {}
        self.synced_animation_db = {}
        self.indices = {}
        self.left_frames = {}
        self.matchable_frames = {}
        self.matching_feature_means = {}
        self.matching_feature_stds = {}
        self.keyjoints_to_use = {}
        self.matching_features_to_use = {}
        self.looping_tags = set()
        self.partwise_tags = set()
        self.hsi_tags = set()
        self.hhi_tags = set()

        # dispatcher for calculating matching features (runtime)
        self.mf_calculators = {
            "keyjoints": self._calc_mf_keyjoints,
            "future_positions": self._calc_mf_future_positions,
            "future_directions": self._calc_mf_future_directions,
            "relative_position": self._calc_mf_relative_position,
            "relative_velocity": self._calc_mf_relative_velocity,
            "relative_direction": self._calc_mf_relative_direction,
            "root_height": self._calc_mf_root_height,
        }

        # dispatcher for pre-calculating matching features (DB creation)
        self.mf_pre_calculators = {
            "keyjoints": self._pre_calc_mf_keyjoints,
            "future_positions": self._pre_calc_mf_future_positions,
            "future_directions": self._pre_calc_mf_future_directions,
            "relative_position": self._pre_calc_mf_relative_position,
            "relative_velocity": self._pre_calc_mf_relative_velocity,
            "relative_direction": self._pre_calc_mf_relative_direction,
            "root_height": self._pre_calc_mf_root_height,
        }

        # load or create motion db
        if cfg.motion_db.recreate_db or not self.db_save_file.exists():
            for tag in tqdm(self.tags, desc="Creating motion DB"):
                self.create_motion_db(tag)
            torch.save(
                {
                    "matching_db": self.matching_db,
                    "animation_db": self.animation_db,
                    "synced_animation_db": self.synced_animation_db,
                    "indices": self.indices,
                    "left_frames": self.left_frames,
                    "matchable_frames": self.matchable_frames,
                    "matching_feature_means": self.matching_feature_means,
                    "matching_feature_stds": self.matching_feature_stds,
                    "keyjoints_to_use": self.keyjoints_to_use,
                    "matching_features_to_use": self.matching_features_to_use,
                    "tags": self.tags,
                    "looping_tags": self.looping_tags,
                    "partwise_tags": self.partwise_tags,
                    "hsi_tags": self.hsi_tags,
                    "hhi_tags": self.hhi_tags,
                },
                self.db_save_file,
            )
        else:
            print(f"Loading motion DB from {self.db_save_file}")
            db = torch.load(
                self.db_save_file,
                map_location=self.device,
                weights_only=False,
            )
            self.matching_db = db["matching_db"]
            self.animation_db = db["animation_db"]
            self.synced_animation_db = db["synced_animation_db"]
            self.indices = db["indices"]
            self.left_frames = db["left_frames"]
            self.matchable_frames = db["matchable_frames"]
            self.matching_feature_means = db["matching_feature_means"]
            self.matching_feature_stds = db["matching_feature_stds"]
            self.keyjoints_to_use = db["keyjoints_to_use"]
            self.matching_features_to_use = db["matching_features_to_use"]
            self.tags = db["tags"]
            self.looping_tags = db["looping_tags"]
            self.partwise_tags = db["partwise_tags"]
            self.hsi_tags = db["hsi_tags"]
            self.hhi_tags = db["hhi_tags"]

    def search_best_matching_frame(self, character: Character, tag):
        matching_feature = self._calc_matching_feature(character, tag)
        matching_feature = (matching_feature - self.matching_feature_means[tag]) / self.matching_feature_stds[tag]

        candidate_mask = self.matchable_frames[tag]
        candidate_features = self.matching_db[tag][candidate_mask]
        candidate_indices = self.indices[tag][candidate_mask]
        feature_distances = (candidate_features - matching_feature).pow(2).sum(dim=1)
        new_fId = candidate_indices[feature_distances.argmin()].item()
        return new_fId

    def get_pose_feature(self, fId, tag):
        return self.animation_db[tag][fId]

    def get_synchronized_pose_feature(self, fId, tag):
        return self.synced_animation_db[tag][fId]

    def get_matching_feature(self, fId, tag, target_feature_type=None):
        mf = self.matching_db[tag][fId] * self.matching_feature_stds[tag] + self.matching_feature_means[tag]
        if target_feature_type is None:
            return mf.unsqueeze(0)

        feature_range = self.get_feature_range(tag, target_feature_type)
        return mf[feature_range].unsqueeze(0)

    @lru_cache(maxsize=32)
    def get_feature_range(self, tag: str, target_feature_type: str):
        idx = 0
        used_mf_types = self.matching_features_to_use[tag]
        used_keyjoints = self.keyjoints_to_use[tag]
        for mf_type in used_mf_types:
            if mf_type == target_feature_type:
                break

            if mf_type == "keyjoints":
                idx += 6 * len(used_keyjoints)
            elif mf_type == "future_positions":
                idx += 2 * self.num_future_offsets
            elif mf_type == "future_directions":
                idx += 2 * self.num_future_offsets
            elif mf_type == "relative_position":
                idx += 2
            elif mf_type == "relative_velocity":
                idx += 2
            elif mf_type == "relative_direction":
                idx += 2
            elif mf_type == "root_height":
                idx += 1
            else:
                raise ValueError(f"Cannot reach here: {mf_type}")

        if target_feature_type == "keyjoints":
            return slice(idx, idx + 6 * len(used_keyjoints))
        elif target_feature_type == "future_positions":
            return slice(idx, idx + 2 * self.num_future_offsets)
        elif target_feature_type == "future_directions":
            return slice(idx, idx + 2 * self.num_future_offsets)
        elif target_feature_type == "relative_position":
            return slice(idx, idx + 2)
        elif target_feature_type == "relative_velocity":
            return slice(idx, idx + 2)
        elif target_feature_type == "relative_direction":
            return slice(idx, idx + 2)
        elif target_feature_type == "root_height":
            return slice(idx, idx + 1)
        else:
            raise ValueError(f"Wrong target_feature_type: {target_feature_type}")

    def is_end_of_motion(self, fId, tag):
        if fId >= len(self.left_frames[tag]):  # FIXME: hacky..
            return True
        return self.left_frames[tag][fId] == 0

    def create_motion_db(self, tag):
        """Create motion database for the given tag."""

        tag_cfg_file = self.db_config_dir / f"{tag}.yaml"
        tag_cfg = yaml.safe_load(tag_cfg_file.open("r"))

        # skip HHI tags if Inter-X dataset is not available
        if tag_cfg["human_human_interaction"] and not inter_x._BASE_DIR.exists():
            self.tags.remove(tag)
            return

        # initialize motion DB for the tag
        matching_feature_dim = self._calculate_matching_feature_dim(tag_cfg)
        self.matching_db[tag] = torch.empty((0, matching_feature_dim), device=self.device)
        self.animation_db[tag] = torch.empty((0, self.cfg.motion_matching.pose_feature_dim), device=self.device)
        self.synced_animation_db[tag] = torch.empty((0, self.cfg.motion_matching.pose_feature_dim), device=self.device)
        self.left_frames[tag] = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.matchable_frames[tag] = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.keyjoints_to_use[tag] = smpl.get_joint_indices(tag_cfg["keyjoints"], device=self.device)
        self.matching_features_to_use[tag] = tag_cfg["matching_features"]

        if tag_cfg["looping"]:
            self.looping_tags.add(tag)
        if tag_cfg["human_human_interaction"]:
            self.hhi_tags.add(tag)
        if tag_cfg["human_scene_interaction"]:
            self.hsi_tags.add(tag)
        if tag.endswith("_partwise"):
            self.partwise_tags.add(tag.replace("_partwise", ""))

        # add motions to the database
        for motion_info in tqdm(
            tag_cfg["motions"],
            desc=f"Adding motions to {tag}",
            leave=False,
        ):
            self.add_motion_to_db(motion_info, tag)

        # calculate mean and std and normalize matching features
        self.matching_feature_means[tag] = self.matching_db[tag].mean(dim=0)
        self.matching_feature_stds[tag] = self.matching_db[tag].std(dim=0)
        self.matching_feature_stds[tag][
            self.matching_feature_stds[tag] == 0
        ] = 1.0  # to prevent the nan at the localized root position
        self.matching_db[tag] = (self.matching_db[tag] - self.matching_feature_means[tag]) / self.matching_feature_stds[
            tag
        ]
        self.indices[tag] = torch.arange(len(self.matching_db[tag]), device=self.device)

    def add_motion_to_db(self, motion_info, tag):
        assert len(self.matching_db[tag]) == len(self.animation_db[tag]) == len(self.matchable_frames[tag])
        dataset = motion_info["dataset"]
        motion_file = motion_info["motion_file"]
        start_frame = motion_info["start_frame"]
        end_frame = motion_info["end_frame"]
        match_start_frame = motion_info["match_start_frame"]
        match_end_frame = motion_info["match_end_frame"]

        # get body params
        body_params, reactor_body_params = self._get_body_params(dataset, motion_file)
        num_frames, framerate = self._get_motion_infos(dataset, motion_file)

        # downsampling
        downsample_rate = round(framerate / self.target_framerate)
        body_params = {k: v[start_frame : end_frame + 1] for k, v in body_params.items()}  # fmt: skip
        body_params = {k: v[::downsample_rate] for k, v in body_params.items()}
        if reactor_body_params is not None:
            reactor_body_params = {k: v[start_frame : end_frame + 1] for k, v in reactor_body_params.items()}  # fmt: skip
            reactor_body_params = {k: v[::downsample_rate] for k, v in reactor_body_params.items()}  # fmt: skip
        num_frames = smpl.get_num_frames(body_params)
        if num_frames <= 30:
            return

        match_start_frame = (match_start_frame - start_frame) // downsample_rate
        match_end_frame = (match_end_frame - start_frame) // downsample_rate

        # get matching features
        matching_features, T_ = self._pre_calc_matching_features(
            body_params,
            reactor_body_params,
            motion_info,
            tag,
        )

        # get pose features
        pose_features = mm.get_pose_features(body_params, self.dt)
        if reactor_body_params is not None:
            synced_pose_features = mm.get_pose_features(reactor_body_params, self.dt)

        # get left_frames and matchable_frames
        left_frames = torch.arange(T_ - 1, -1, -1, device=self.device, dtype=torch.int32)  # fmt: skip
        matchable_frames = torch.zeros((T_,), dtype=torch.bool, device=self.device)
        matchable_frames[match_start_frame:match_end_frame] = True
        matchable_frames[left_frames < self.motion_update_frequency] = False

        # add to motion db
        self.matching_db[tag] = torch.cat([self.matching_db[tag], matching_features], dim=0)
        self.animation_db[tag] = torch.cat([self.animation_db[tag], pose_features[:T_]], dim=0)
        if reactor_body_params is not None:
            self.synced_animation_db[tag] = torch.cat([self.synced_animation_db[tag], synced_pose_features[:T_]], dim=0)
        self.left_frames[tag] = torch.cat([self.left_frames[tag], left_frames], dim=0)
        self.matchable_frames[tag] = torch.cat([self.matchable_frames[tag], matchable_frames], dim=0)

    def _calculate_matching_feature_dim(self, tag_cfg):
        matching_features_to_use = tag_cfg["matching_features"]
        keyjoints_to_use = tag_cfg["keyjoints"]

        matching_feature_dim = 0
        for matching_feature in matching_features_to_use:
            if matching_feature == "keyjoints":
                matching_feature_dim += 6 * len(keyjoints_to_use)
            elif matching_feature == "future_positions":
                matching_feature_dim += 2 * self.num_future_offsets
            elif matching_feature == "future_directions":
                matching_feature_dim += 2 * self.num_future_offsets
            elif matching_feature == "relative_position":
                matching_feature_dim += 2
            elif matching_feature == "relative_velocity":
                matching_feature_dim += 2
            elif matching_feature == "relative_direction":
                matching_feature_dim += 2
            elif matching_feature == "root_height":
                matching_feature_dim += 1
            else:
                raise ValueError(f"Unknown matching feature: {matching_feature}")
        return matching_feature_dim

    def _get_body_params(self, dataset, motion_file):
        assert dataset in ["amass", "samp", "inter_x", "mixamo"]
        if dataset in ["amass", "mixamo"]:
            motion_data = amass.get_motion(motion_file)
            body_params = amass.construct_body_params(motion_data)
            reactor_body_params = None
        elif dataset == "samp":
            motion_data = samp.get_motion(motion_file)
            body_params = samp.construct_body_params(motion_data)
            reactor_body_params = None
        elif dataset == "inter_x":
            motion_data = inter_x.get_motions(motion_file)
            body_params = inter_x.construct_body_params(motion_data)
            body_params, reactor_body_params = body_params
        return body_params, reactor_body_params

    def _get_motion_infos(self, dataset, motion_file):
        assert dataset in ["amass", "samp", "inter_x", "mixamo"]
        if dataset in ["amass", "mixamo"]:
            motion_data = amass.get_motion(motion_file)
            num_frames, framerate, _ = amass.get_motion_info(motion_data)
        elif dataset == "samp":
            motion_data = samp.get_motion(motion_file)
            num_frames, framerate, _ = samp.get_motion_info(motion_data)
        elif dataset == "inter_x":
            motion_data = inter_x.get_motions(motion_file)
            num_frames, framerate, _ = inter_x.get_motion_infos(motion_data)
        return num_frames, framerate

    def _calc_matching_feature(self, character: Character, tag):
        # last two frames for velocity calculation
        body_params = mm.to_dict(character.body_params[-2:])
        facing_directions = smpl.get_facing_directions(body_params)
        character_positions = smpl.get_character_positions(body_params)
        character_rotations = smpl.get_character_rotations(facing_directions)

        matching_features = []
        for mf_type in self.matching_features_to_use[tag]:
            mf = self.mf_calculators[mf_type](
                character,
                body_params,
                character_positions,
                character_rotations,
                tag,
            )
            matching_features.append(mf)
        return torch.cat(matching_features, dim=1)

    def _pre_calc_matching_features(self, body_params, reactor_body_params, motion_info, tag):
        facing_directions = smpl.get_facing_directions(body_params)
        character_positions = smpl.get_character_positions(body_params)
        character_rotations = smpl.get_character_rotations(facing_directions)

        matching_features = []
        for mf_type in self.matching_features_to_use[tag]:
            mf = self.mf_pre_calculators[mf_type](
                body_params,
                reactor_body_params,
                character_positions,
                character_rotations,
                motion_info,
                tag,
            )
            matching_features.append(mf)

        # when future_positions or future_directions are used, we need to truncate
        # the other matching features to the same length
        T_ = min([mf.shape[0] for mf in matching_features])
        matching_features = [mf[:T_] for mf in matching_features]
        matching_features = torch.cat(matching_features, dim=1)
        return matching_features, T_

    def _calc_mf_keyjoints(
        self,
        character,
        body_params,
        character_positions,
        character_rotations,
        tag,
    ):
        num_frames = smpl.get_num_frames(body_params)
        keyjoint_positions, keyjoint_velocities = mm.get_keyjoint_features(
            body_params,
            self.keyjoints_to_use[tag],
            character_positions,
            character_rotations,
        )
        keyjoint_features = torch.cat([keyjoint_positions, keyjoint_velocities], dim=-1)
        keyjoint_features = keyjoint_features.reshape(num_frames, -1)
        return keyjoint_features[-1:]

    def _pre_calc_mf_keyjoints(
        self,
        body_params,
        reactor_body_params,
        character_positions,
        character_rotations,
        motion_info,
        tag,
    ):
        num_frames = smpl.get_num_frames(body_params)
        keyjoint_positions, keyjoint_velocities = mm.get_keyjoint_features(
            body_params,
            self.keyjoints_to_use[tag],
            character_positions,
            character_rotations,
        )
        keyjoint_features = torch.cat([keyjoint_positions, keyjoint_velocities], dim=-1)
        keyjoint_features = keyjoint_features.reshape(num_frames, -1)
        return keyjoint_features

    def _calc_mf_future_positions(
        self,
        character,
        body_params,
        character_positions,
        character_rotations,
        tag,
    ):
        future_positions = character.future_positions
        future_positions = mm.to_local(future_positions, character_positions[-1:], character_rotations[-1:])
        future_positions = future_positions[..., :2].reshape(1, -1)
        return future_positions

    def _pre_calc_mf_future_positions(
        self,
        body_params,
        reactor_body_params,
        character_positions,
        character_rotations,
        motion_info,
        tag,
    ):
        future_positions = mm.get_future_positions(
            body_params,
            self.future_offsets,
            character_positions,
            character_rotations,
        )
        num_frames = future_positions.shape[0]
        future_positions = future_positions[..., :2].reshape(num_frames, -1)
        return future_positions

    def _calc_mf_future_directions(
        self,
        character,
        body_params,
        character_positions,
        character_rotations,
        tag,
    ):
        future_directions = character.future_facing_directions
        future_directions = mm.to_local(future_directions, character_rotations[-1:])
        future_directions = future_directions[..., :2].reshape(1, -1)
        return future_directions

    def _pre_calc_mf_future_directions(
        self,
        body_params,
        reactor_body_params,
        character_positions,
        character_rotations,
        motion_info,
        tag,
    ):
        facing_directions = smpl.get_facing_directions(body_params)
        future_directions = mm.get_future_facing_directions(
            body_params,
            self.future_offsets,
            facing_directions,
            character_rotations,
        )
        num_frames = future_directions.shape[0]
        future_directions = future_directions[..., :2].reshape(num_frames, -1)
        return future_directions

    def _calc_mf_relative_position(
        self,
        character,
        body_params,
        character_positions,
        character_rotations,
        tag,
    ):
        if tag in self.hsi_tags:
            target_position = character.target_root_position
        else:
            target_position = smpl.get_character_positions(character.interactee.body_params[-1:])
        relative_position = mm.to_local(
            target_position,
            character_positions[-1:],
            character_rotations[-1:],
        )
        return relative_position[:, :2]

    def _pre_calc_mf_relative_position(
        self,
        body_params,
        reactor_body_params,
        character_positions,
        character_rotations,
        motion_info,
        tag,
    ):
        if tag in self.hsi_tags:
            target_positions = torch.tensor(
                motion_info["target_root_position"],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)
        else:
            target_positions = smpl.get_character_positions(reactor_body_params)
        relative_positions = mm.to_local(
            target_positions,
            character_positions,
            character_rotations,
        )
        return relative_positions[:, :2]

    def _calc_mf_relative_velocity(
        self,
        character,
        body_params,
        character_positions,
        character_rotations,
        tag,
    ):
        if tag in self.hsi_tags:
            target_positions = character.target_root_position.tile(2, 1)
        else:
            target_positions = smpl.get_character_positions(character.interactee.body_params[-2:])
        relative_positions = target_positions - character_positions
        relative_velocities = torch.diff(relative_positions, dim=0, prepend=relative_positions[:1]) / self.dt
        relative_velocities = mm.to_local(relative_velocities, character_rotations)
        return relative_velocities[-1:, :2]

    def _pre_calc_mf_relative_velocity(
        self,
        body_params,
        reactor_body_params,
        character_positions,
        character_rotations,
        motion_info,
        tag,
    ):
        if tag in self.hsi_tags:
            num_frames = smpl.get_num_frames(body_params)
            target_positions = (
                torch.tensor(
                    motion_info["target_root_position"],
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 3)
                .tile(num_frames, 1)
            )
        else:
            target_positions = smpl.get_character_positions(reactor_body_params)
        relative_positions = target_positions - character_positions
        relative_velocities = torch.diff(relative_positions, dim=0, prepend=relative_positions[:1]) / self.dt
        relative_velocities = mm.to_local(relative_velocities, character_rotations)
        return relative_velocities[:, :2]

    def _calc_mf_relative_direction(
        self,
        character,
        body_params,
        character_positions,
        character_rotations,
        tag,
    ):
        if tag in self.hsi_tags:
            target_direction = character.target_facing_direction
        else:
            target_direction = smpl.get_facing_directions(character.interactee.body_params[-1:])
        relative_direction = mm.to_local(target_direction, character_rotations[-1:])
        return relative_direction[:, :2]

    def _pre_calc_mf_relative_direction(
        self,
        body_params,
        reactor_body_params,
        character_positions,
        character_rotations,
        motion_info,
        tag,
    ):
        if tag in self.hsi_tags:
            target_direction = torch.tensor(
                motion_info["target_direction"],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)
        else:
            target_direction = smpl.get_facing_directions(reactor_body_params)
        relative_direction = mm.to_local(target_direction, character_rotations)
        return relative_direction[:, :2]

    def _calc_mf_root_height(
        self,
        character,
        body_params,
        character_positions,
        character_rotations,
        tag,
    ):
        target_height = character.target_root_position[:, 2:]
        return target_height

    def _pre_calc_mf_root_height(
        self,
        body_params,
        reactor_body_params,
        character_positions,
        character_rotations,
        motion_info,
        tag,
    ):
        num_frames = smpl.get_num_frames(body_params)
        target_height = torch.tensor(
            motion_info["target_root_position"],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3)
        target_height = target_height[:, 2:].tile(num_frames, 1)
        return target_height


if __name__ == "__main__":
    from src.utils import misc

    cfg = misc.load_cfg("./configs/demo.yaml", read_only=False)
    cfg.motion_db.recreate_db = True
    db = MotionDB(cfg)
