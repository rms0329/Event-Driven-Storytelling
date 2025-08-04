import json
import os
from pathlib import Path

import huggingface_hub
from dotenv import load_dotenv

load_dotenv()


def download_hssd():
    """
    Selectively download the 3D models from the HSSD dataset.
    Only the objects used in our test scenes will be downloaded.
    """

    # login to Hugging Face Hub
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_HUB_TOKEN not found in the .env file.")
    huggingface_hub.login(token)

    # download the models
    save_dir = Path("./data/HSSD")
    save_dir.mkdir(parents=True, exist_ok=True)
    for scene_name in ["House", "Office", "Restaurant"]:
        scene_cfg_file = Path(f"configs/scenes/{scene_name}/scene_cfg.json")
        scene_cfg = json.load(scene_cfg_file.open())

        for _, data in scene_cfg.items():
            model_id = data["model_id"]
            filename = f"objects/{model_id[0]}/{model_id}.glb"
            obj_file = save_dir / filename
            if not obj_file.exists():
                huggingface_hub.hf_hub_download(
                    repo_id="hssd/hssd-models",
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(save_dir),
                )

    print("HSSD dataset downloaded successfully.")


if __name__ == "__main__":
    download_hssd()
