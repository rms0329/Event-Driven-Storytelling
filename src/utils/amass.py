import random
from pathlib import Path

import numpy as np
import torch

from . import smpl

# Total number of frames: 22061680
# Total number of files: 15805
# Total number of non-error files: 15653
# Total number of error files: 152

BASE_DIR = Path("data/AMASS/SMPL-H-G")
_AMASS_SUBSETS = {
    "ACCAD",
    "BMLhandball",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "CNRS",
    "DFaust",
    "DanceDB",
    "EKUT",
    "EyesJapanDataset",
    "GRAB",
    "HDM05",
    "HUMAN4D",
    "HumanEva",
    "KIT",
    "MOYO",
    "MoSh",
    "PosePrior",
    "SFU",
    "SOMA",
    "SSM",
    "TCDHands",
    "TotalCapture",
    "Transitions",
    "WEIZMANN",
}


MANUAL_FRAMERATE = {
    "SSM": 60.0,
    "DFaust": 60.0,
}


def get_motion_datasets(split="all", as_path=False):
    # https://github.com/nghorbani/amass/blob/master/notebooks/02-AMASS_DNN.ipynb
    validation_datasets = ["HumanEva", "HDM05", "SFU", "MoSh"]
    test_datasets = ["Transitions", "SSM"]
    train_datasets = [
        p.name for p in BASE_DIR.iterdir() if p.is_dir() and p.name not in validation_datasets + test_datasets
    ]
    if as_path:
        validation_datasets = [BASE_DIR / p for p in validation_datasets]
        test_datasets = [BASE_DIR / p for p in test_datasets]
        train_datasets = [BASE_DIR / p for p in train_datasets]

    if split == "all":
        return train_datasets + validation_datasets + test_datasets
    elif split == "train":
        return train_datasets
    elif split == "val":
        return validation_datasets
    elif split == "test":
        return test_datasets
    else:
        raise ValueError(f"Invalid split '{split}'")


def get_random_motion(with_source=False, split=None):
    if split is None:
        motion_datasets = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    else:
        motion_datasets = [p for p in BASE_DIR.iterdir() if p.is_dir() and p.name in get_motion_datasets(split)]
    motion_dataset = random.choice(motion_datasets)
    sub_dirs = [p for p in motion_dataset.iterdir() if p.is_dir()]
    sub_dir = random.choice(sub_dirs)
    npz_files = list(sub_dir.glob("[!.]*.npz"))
    npz_file = random.choice(npz_files)
    if with_source:
        print(f"Loading '{'/'.join(npz_file.parts[-3:])}'")
        return np.load(npz_file), npz_file
    return np.load(npz_file)


def get_motion(motion_path, with_source=False):
    motion_path = Path(motion_path)

    if any(part in _AMASS_SUBSETS for part in motion_path.parts):
        if BASE_DIR in motion_path.parents:
            npz_file = motion_path
        else:
            npz_file = BASE_DIR / motion_path
    else:
        npz_file = motion_path

    if with_source:
        return np.load(npz_file), npz_file
    return np.load(npz_file)


def get_motion_info(motion_data):
    """Returns the number of frames, framerate, and gender"""
    num_frames = len(motion_data["poses"])

    framerate = (
        float(motion_data["mocap_framerate"])
        if "mocap_framerate" in motion_data
        else float(motion_data["mocap_frame_rate"])
    )
    dataset_name = Path(motion_data.fid.name).parts[-3]
    if dataset_name in MANUAL_FRAMERATE:
        framerate = MANUAL_FRAMERATE[dataset_name]

    gender = np.array2string(motion_data["gender"]).replace("'", "").strip().lower().lstrip("b")
    assert gender in [
        "male",
        "female",
        "neutral",
    ], f"Invalid gender '{gender}' found in motion data."
    return num_frames, framerate, gender


def iterate_motions(with_source=False, split=None):
    if split is None:
        npz_files = BASE_DIR.rglob("[!.]*.npz")
        for npz_file in npz_files:
            if with_source:
                yield np.load(npz_file), npz_file
            else:
                yield np.load(npz_file)

    else:
        datasets = get_motion_datasets(split)
        for dataset in datasets:
            npz_files = (BASE_DIR / dataset).rglob("[!.]*.npz")
            for npz_file in npz_files:
                if with_source:
                    yield np.load(npz_file), npz_file
                else:
                    yield np.load(npz_file)


def construct_body_params(
    motion_data,
    device="cuda",
    use_zero_betas=True,
    num_betas=10,
    gender=None,
):
    num_frames, framerate, original_gender = get_motion_info(motion_data)
    if gender is None:
        gender = original_gender

    body_params = {
        "transl": torch.tensor(motion_data["trans"], dtype=torch.float32, device=device),
        "global_orient": torch.tensor(motion_data["poses"][:, :3], dtype=torch.float32, device=device),
        "body_pose": torch.tensor(motion_data["poses"][:, 3:66], dtype=torch.float32, device=device),
        "left_hand_pose": torch.tensor(motion_data["poses"][:, 66:111], dtype=torch.float32, device=device),
        "right_hand_pose": torch.tensor(motion_data["poses"][:, 111:156], dtype=torch.float32, device=device),
        "betas": torch.tensor(
            np.repeat(motion_data["betas"][:num_betas][np.newaxis], repeats=num_frames, axis=0),
            dtype=torch.float32,
            device=device,
        ),
    }
    if use_zero_betas:
        body_params["betas"][...] = 0.0
    if use_zero_betas or gender != original_gender:  # adjust root height
        bm = smpl.load_body_model(device=device, gender=gender)
        first_frame_mesh = bm(**{k: v[:1] for k, v in body_params.items()}).vertices[0]
        body_params["transl"][:, 2] -= first_frame_mesh[:, 2].min()

    return body_params
