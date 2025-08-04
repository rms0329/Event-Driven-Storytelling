import numpy as np

from src.scene.scene import Scene
from src.utils import misc
from src.utils.polyscope import PolyscopeApp


def main():
    cfg = misc.load_cfg("./configs/demo.yaml")
    scene = Scene(cfg.scene_name, cfg)

    app = PolyscopeApp(show_axis=True)
    for obj in scene.objects:
        color = np.random.rand(3)
        app.register_trimesh(f"{obj}", obj.mesh)
        app.register_bounding_box(f"{obj}_bbox", obj.bbox, color=color, radius=0.03)
        if obj.has_orientation:
            app.register_arrow(
                f"{obj}_orientation",
                obj.center,
                obj.center + obj.orientation,
                color=color,
                radius=0.01,
                length=0.4,
            )
    app.run()


if __name__ == "__main__":
    main()
