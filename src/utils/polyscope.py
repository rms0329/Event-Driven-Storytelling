import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh
import trimesh.creation
import trimesh.visual
from polyscope.curve_network import CurveNetwork
from polyscope.surface_mesh import SurfaceMesh


class PolyscopeApp:
    def __init__(
        self,
        app_name="PolyscopeApp",
        framerate=30,
        ground_plane_mode="shadow_only",
        up_dir="z_up",
        show_axis=False,
    ) -> None:
        ps.set_program_name(app_name)
        ps.set_up_dir(up_dir)
        ps.set_max_fps(int(framerate))
        ps.set_ground_plane_mode(ground_plane_mode)
        ps.set_give_focus_on_show(True)
        ps.set_enable_vsync(False)
        ps.init()
        if show_axis:
            axis = trimesh.creation.axis(origin_size=0.01, axis_length=1, axis_radius=0.02)
            self._axis = self.register_trimesh("axis", axis)
        self.gui = psim

    def register_mesh(self, name, vertices, faces, color=None, **kwargs):
        return ps.register_surface_mesh(name, vertices, faces, color=color, smooth_shade=True, **kwargs)

    def register_trimesh(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        color=None,
        material="clay",
        back_face_policy="cull",
        use_original_color=True,
        smooth_shade=True,
        enabled=True,
    ) -> SurfaceMesh:
        ps_mesh = ps.register_surface_mesh(
            name,
            mesh.vertices,
            mesh.faces,
            material=material,
            smooth_shade=smooth_shade,
            enabled=enabled,
        )
        ps_mesh.set_back_face_policy(back_face_policy)

        if color is not None:
            ps_mesh.set_color(color)

        elif use_original_color:
            if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                uv = mesh.visual.uv
                if isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
                    img = mesh.visual.material.baseColorTexture
                else:
                    img = mesh.visual.material.image

                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = np.array(img).astype(np.float32) / 255.0
                ps_mesh.add_parameterization_quantity("uv", uv, defined_on="vertices", enabled=True)
                ps_mesh.add_color_quantity("texture", img, defined_on="texture", param_name="uv", enabled=True)
            else:
                colors = mesh.visual.vertex_colors
                if colors.ndim == 1:
                    num_vertices = mesh.vertices.shape[0]
                    colors = np.tile(colors, (num_vertices, 1))
                colors = colors[:, :3].astype(np.float32) / 255.0
                ps_mesh.add_color_quantity("colors", colors, enabled=True)

        return ps_mesh

    def register_bounding_box(
        self, name: str, bbox: np.ndarray, radius=0.001, color=None, enabled=True
    ) -> CurveNetwork:
        """
        bbox: 8x3 numpy array.
        Each row is a vertex of the bounding box.
        The first 4 vertices are the bottom face of the bounding box and the last 4 vertices are the top face of the bounding box.
        """
        assert bbox.shape == (8, 3)
        ps_bbox = ps.register_curve_network(
            name,
            bbox,
            np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [0, 3],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [4, 7],
                ]
            ),
            color=color,
            enabled=enabled,
        )
        ps_bbox.set_radius(radius, relative=False)
        return ps_bbox

    def register_arrow(
        self, name: str, start: np.ndarray, end: np.ndarray, radius=0.001, length=1.0, color=None, enabled=True
    ):
        direction = end - start
        direction /= np.linalg.norm(direction)

        ps_arrow = ps.register_point_cloud(name, start.reshape(1, 3), radius=0.0, color=color, enabled=enabled)
        ps_arrow.add_vector_quantity(
            "direction",
            direction.reshape(1, 3),
            radius=radius,
            length=length,
            color=color,
            vectortype="ambient",  # ambient vectors don't get auto-scaled
            enabled=True,
        )
        return ps_arrow

    def error(self, message: str):
        ps.error(message)

    def warning(self, message: str):
        ps.warning(message)

    # https://github.com/ocornut/imgui/blob/4f9ba19e520bea478f5cb654d37ef45e6404bd52/imgui.h#L1322
    def is_key_pressed(self, key: str) -> bool:
        imgui_key = getattr(psim, f"ImGuiKey_{key}")
        return psim.IsKeyPressed(psim.GetKeyIndex(imgui_key))

    def is_key_down(self, key: str) -> bool:
        imgui_key = getattr(psim, f"ImGuiKey_{key}")
        return psim.IsKeyDown(psim.GetKeyIndex(imgui_key))

    def _user_callback(self):
        self.ui_callback()
        self.callback()
        if self.is_key_pressed("Escape"):
            ps.unshow()

    def ui_callback(self):
        pass

    def callback(self):
        pass

    def reset(self):
        ps.remove_all_structures()
        if hasattr(self, "_axis"):
            axis = trimesh.creation.axis(origin_size=0.01, axis_length=1, axis_radius=0.02)
            self._axis = self.register_trimesh("axis", axis)

    def run(self):
        ps.set_user_callback(self._user_callback)
        ps.show()
