"""Optional Panda3D renderer for the warehouse verse's interactive preview.

The simulation remains independent of Panda3D.  This module is loaded only
when an RGB frame is requested, and the verse falls back to its local software
rasterizer if Panda3D or an offscreen graphics context is unavailable.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

_renderer: Optional["PandaWarehouseRenderer"] = None


class PandaWarehouseRenderer:
    """Persistent, anti-aliased offscreen renderer used by the pygame viewer."""

    def __init__(self, width: int, height: int) -> None:
        from panda3d.core import loadPrcFileData

        loadPrcFileData(
            "",
            "\n".join(
                (
                    "window-type offscreen",
                    f"win-size {int(width)} {int(height)}",
                    "sync-video false",
                    "audio-library-name null",
                    "notify-level error",
                    "framebuffer-multisample 1",
                    "multisamples 8",
                )
            ),
        )

        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import AmbientLight, DirectionalLight, NodePath, Point3, Vec4

        self.width = max(16, int(width))
        self.height = max(16, int(height))
        self._Point3 = Point3
        self.base = ShowBase(windowType="offscreen")
        if self.base.win is None:
            raise RuntimeError("Panda3D could not create an offscreen graphics buffer")
        self.base.disableMouse()
        self.base.setBackgroundColor(0.19, 0.23, 0.27, 1.0)
        self.base.camLens.setFov(48.0)
        self.base.camLens.setNearFar(0.08, 100.0)

        self._materials = {}
        self._box_template = self.base.loader.loadModel("models/box")
        self._sphere_template = self._make_sphere_template(NodePath)
        self._world = self.base.render.attachNewNode("warehouse_world")
        self._layout: Optional[Tuple[Any, ...]] = None
        self._arena: Optional[float] = None
        self._goal: Optional[Tuple[float, ...]] = None
        self._robot = None

        ambient = AmbientLight("warehouse_ambient")
        ambient.setColor(Vec4(0.68, 0.70, 0.72, 1.0))
        self._ambient_path = self.base.render.attachNewNode(ambient)
        self.base.render.setLight(self._ambient_path)

        sunlight = DirectionalLight("warehouse_sunlight")
        sunlight.setColor(Vec4(0.94, 0.91, 0.84, 1.0))
        # Keep the broad light shadow-free until the orthographic shadow camera
        # can be tightly fitted to the active warehouse bounds.
        self._sun_path = self.base.render.attachNewNode(sunlight)
        self._sun_path.setPos(0.0, -4.0, 12.0)
        self._sun_path.lookAt(Point3(0.0, 0.0, 0.0))
        self.base.render.setLight(self._sun_path)
        self.base.render.setShaderAuto()

    @staticmethod
    def _make_sphere_template(node_path_type):
        from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter

        rings, sectors = 12, 18
        data = GeomVertexData("warehouse_sphere", GeomVertexFormat.getV3n3(), Geom.UHStatic)
        vertices = GeomVertexWriter(data, "vertex")
        normals = GeomVertexWriter(data, "normal")
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for sector in range(sectors + 1):
                theta = 2.0 * math.pi * sector / sectors
                point = (
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi),
                )
                vertices.addData3(*point)
                normals.addData3(*point)

        triangles = GeomTriangles(Geom.UHStatic)
        stride = sectors + 1
        for ring in range(rings):
            for sector in range(sectors):
                current = ring * stride + sector
                around = current + 1
                below = current + stride
                below_around = below + 1
                triangles.addVertices(current, below, around)
                triangles.addVertices(around, below, below_around)
        triangles.closePrimitive()
        geometry = Geom(data)
        geometry.addPrimitive(triangles)
        node = GeomNode("warehouse_unit_sphere")
        node.addGeom(geometry)
        return node_path_type(node)

    def _material(self, color: Sequence[int]):
        from panda3d.core import Material, Vec4

        rgb = tuple(max(0.0, min(1.0, component / 255.0)) for component in color)
        key = rgb
        cached = self._materials.get(key)
        if cached is not None:
            return cached

        material = Material(f"warehouse_material_{len(self._materials)}")
        material.setDiffuse(Vec4(*rgb, 1.0))
        material.setAmbient(Vec4(*(component * 0.64 for component in rgb), 1.0))
        material.setSpecular(Vec4(0.08, 0.08, 0.08, 1.0))
        material.setEmission(Vec4(*(component * 0.025 for component in rgb), 1.0))
        material.setShininess(42.0)
        self._materials[key] = material
        return material

    def _box(
        self,
        parent,
        name: str,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        color: Sequence[int],
    ):
        node = parent.attachNewNode(name)
        model = self._box_template.instanceTo(node)
        # Panda's bundled unit box spans [0, 1] on each axis. Shift its mesh
        # origin to the center so position/size match the verse's box helpers.
        model.setPos(-0.5, -0.5, -0.5)
        node.setTextureOff(1)
        node.setMaterial(self._material(color), 1)
        node.setColor(*(max(0.0, min(1.0, channel / 255.0)) for channel in color), 1.0)
        node.setPos(float(position[0]), float(position[1]), float(position[2]))
        node.setScale(float(size[0]), float(size[1]), float(size[2]))
        return node

    def _sphere(
        self,
        parent,
        name: str,
        position: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        color: Sequence[int],
    ):
        node = parent.attachNewNode(name)
        self._sphere_template.instanceTo(node)
        node.setTextureOff(1)
        node.setMaterial(self._material(color), 1)
        node.setColor(*(max(0.0, min(1.0, channel / 255.0)) for channel in color), 1.0)
        node.setPos(float(position[0]), float(position[1]), float(position[2]))
        node.setScale(float(scale[0]), float(scale[1]), float(scale[2]))
        return node

    def _cylinder(
        self,
        parent,
        name: str,
        position: Tuple[float, float, float],
        radius: float,
        height: float,
        color: Sequence[int],
    ):
        node = parent.attachNewNode(name)
        # A flattened smooth sphere gives a clean local fallback for a cylinder
        # without relying on Panda3D's optional sample-model bundle.
        self._sphere_template.instanceTo(node)
        node.setTextureOff(1)
        node.setMaterial(self._material(color), 1)
        node.setColor(*(max(0.0, min(1.0, channel / 255.0)) for channel in color), 1.0)
        node.setPos(float(position[0]), float(position[1]), float(position[2]))
        node.setScale(float(radius), float(radius), float(height) * 0.5)
        return node

    def _line(self, name: str, points, color, thickness: float):
        from panda3d.core import LineSegs, Vec4

        lines = LineSegs(name)
        lines.setThickness(float(thickness))
        lines.setColor(Vec4(color[0], color[1], color[2], 1.0))
        for start, end in points:
            lines.moveTo(*start)
            lines.drawTo(*end)
        return self._world.attachNewNode(lines.create())

    def _ground_plane(self, extent: float, color: Sequence[int]):
        from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter

        data = GeomVertexData("warehouse_concrete", GeomVertexFormat.getV3n3(), Geom.UHStatic)
        vertices = GeomVertexWriter(data, "vertex")
        normals = GeomVertexWriter(data, "normal")
        for point in ((-extent, -extent, 0.0), (extent, -extent, 0.0), (extent, extent, 0.0), (-extent, extent, 0.0)):
            vertices.addData3(*point)
            normals.addData3(0.0, 0.0, 1.0)

        triangles = GeomTriangles(Geom.UHStatic)
        triangles.addVertices(0, 1, 2)
        triangles.addVertices(0, 2, 3)
        triangles.closePrimitive()
        geometry = Geom(data)
        geometry.addPrimitive(triangles)
        node = GeomNode("concrete_floor")
        node.addGeom(geometry)
        floor = self._world.attachNewNode(node)
        floor.setMaterial(self._material(color), 1)
        floor.setColor(*(max(0.0, min(1.0, channel / 255.0)) for channel in color), 1.0)
        floor.setTwoSided(True)
        return floor

    def _build_world(self, crates, arena: float, goal) -> None:
        self._world.removeNode()
        self._world = self.base.render.attachNewNode("warehouse_world")
        arena = max(1.0, float(arena))
        self._layout = tuple(crates)
        self._arena = arena
        self._goal = tuple(float(v) for v in goal)

        # Concrete floor and subtle aisle grid.
        self._ground_plane(15.0, (107, 117, 120))
        extent = int(math.floor(arena))
        grid_lines = []
        for coordinate in range(-extent, extent + 1):
            grid_lines.append(((-extent, coordinate, 0.008), (extent, coordinate, 0.008)))
            grid_lines.append(((coordinate, -extent, 0.008), (coordinate, extent, 0.008)))
        self._line("floor_grid", grid_lines, (0.29, 0.36, 0.37), 1.0)
        boundary = []
        for side in (-arena, arena):
            boundary.extend(
                (((-arena, side, 0.014), (arena, side, 0.014)), ((side, -arena, 0.014), (side, arena, 0.014)))
            )
        self._line("floor_boundary", boundary, (0.58, 0.68, 0.65), 2.0)

        # Warehouse shell and modular steel storage rack.
        wall_z = arena + 0.58
        self._box(self._world, "rear_wall", (0.0, wall_z, 2.65), (2.0 * arena + 1.0, 0.18, 5.3), (66, 77, 82))
        for seam in range(-3, 4):
            x = seam * (2.0 * arena + 1.0) / 7.0
            self._box(self._world, f"rear_wall_seam_{seam}", (x, wall_z - 0.1, 2.65), (0.025, 0.025, 5.1), (80, 91, 94))
        rack_z = min(arena - 0.55, 4.72)
        rack_width = min(7.8, arena * 1.42)
        rack_xs = (-rack_width * 0.5, -rack_width / 6.0, rack_width / 6.0, rack_width * 0.5)
        for index, x in enumerate(rack_xs):
            self._box(self._world, f"rack_upright_{index}", (x, rack_z, 1.45), (0.11, 0.14, 2.9), (83, 99, 105))
        for level, shelf_z in enumerate((0.28, 1.34, 2.42)):
            self._box(self._world, f"rack_beam_{level}", (0.0, rack_z - 0.04, shelf_z), (rack_width, 0.22, 0.13), (184, 124, 64))

        pallet_h = 0.18
        for name, cx, cz, width, height, depth, color in crates:
            for index, offset_x in enumerate((-0.34, 0.0, 0.34)):
                self._box(self._world, f"{name}_pallet_runner_{index}", (cx + offset_x * width, cz, 0.055), (width * 0.14, depth * 0.92, 0.11), (132, 91, 54))
            for index, offset_z in enumerate((-0.34, 0.0, 0.34)):
                self._box(self._world, f"{name}_pallet_deck_{index}", (cx, cz + offset_z * depth, 0.135), (width * 0.98, depth * 0.17, 0.05), (166, 119, 75))

            self._box(self._world, name, (cx, cz, pallet_h + height * 0.5), (width, depth, height), color)
            label_y = pallet_h + height * 0.66
            front_z = cz - depth * 0.5 - 0.015
            self._box(self._world, f"{name}_label", (cx + width * 0.13, front_z, label_y), (width * 0.38, 0.024, max(0.16, height * 0.16)), (225, 221, 203))
            for mark, (offset_y, fraction) in enumerate(((0.0, 0.2), (-0.045, 0.34), (0.045, 0.27))):
                self._box(self._world, f"{name}_label_mark_{mark}", (cx + width * 0.13, front_z - 0.014, label_y + offset_y), (width * fraction, 0.012, 0.018), (54, 62, 65))

        support = next((crate for crate in crates if crate[0] == "gray"), None)
        if support is not None:
            _, px, pz, _, support_height, _, _ = support
            package_y = pallet_h + support_height + 0.22
            self._box(self._world, "package", (px, pz, package_y), (0.34, 0.32, 0.28), (194, 151, 91))
            self._box(self._world, "package_label", (px, pz - 0.174, package_y), (0.16, 0.018, 0.11), (224, 216, 191))

        gx, gz, goal_radius = self._goal
        self._cylinder(self._world, "goal_zone_outer", (gx, gz, 0.012), goal_radius + 0.09, 0.024, (40, 75, 63))
        self._cylinder(self._world, "goal_zone_inner", (gx, gz, 0.029), goal_radius, 0.018, (59, 133, 89))
        self._sphere(self._world, "goal_marker", (gx, gz, 0.19), (0.15, 0.15, 0.15), (255, 199, 70))

        self._robot = self._world.attachNewNode("mobile_robot")
        # Low-profile AMR chassis, four rubber wheels, lidar mast, and safety lamps.
        self._box(self._robot, "wheel_base", (0.0, 0.0, 0.16), (0.82, 0.68, 0.16), (43, 50, 53))
        self._box(self._robot, "amber_chassis", (0.0, 0.0, 0.30), (0.78, 0.66, 0.22), (207, 116, 43))
        self._box(self._robot, "front_bumper", (0.0, 0.34, 0.27), (0.62, 0.035, 0.055), (62, 69, 71))
        for x in (-0.42, 0.42):
            for y in (-0.27, 0.27):
                self._sphere(self._robot, f"wheel_{x:+.2f}_{y:+.2f}", (x, y, 0.14), (0.085, 0.13, 0.13), (34, 39, 40))
        self._cylinder(self._robot, "lidar_mount", (0.0, 0.0, 0.49), 0.055, 0.15, (74, 86, 89))
        self._cylinder(self._robot, "lidar_sensor", (0.0, 0.0, 0.60), 0.15, 0.10, (41, 133, 143))
        self._box(self._robot, "heading_arrow", (0.0, 0.21, 0.48), (0.075, 0.28, 0.025), (240, 190, 73))
        for index, x in enumerate((-0.23, 0.23)):
            self._sphere(self._robot, f"headlamp_{index}", (x, 0.37, 0.30), (0.055, 0.035, 0.055), (239, 218, 151))

        self._layout = tuple(crates)
        self._arena = arena

    def render(
        self,
        *,
        robot_x: float,
        robot_z: float,
        yaw: float,
        crates,
        arena: float,
        goal,
        camera_mode: str,
        width: int,
        height: int,
    ) -> np.ndarray:
        layout_key = tuple(crates)
        if self._layout != layout_key or self._arena != float(arena) or self._goal != tuple(goal):
            self._build_world(crates, arena, goal)

        if self.width != int(width) or self.height != int(height):
            self.width, self.height = max(16, int(width)), max(16, int(height))
            self.base.win.setSize(self.width, self.height)

        forward_x = math.sin(float(yaw))
        forward_z = math.cos(float(yaw))
        self._robot.setPos(float(robot_x), float(robot_z), 0.0)
        self._robot.setH(-math.degrees(float(yaw)))

        if camera_mode == "isometric":
            eye = (6.2, -5.2, 8.4)
            target = (0.0, 1.0, 0.65)
        else:
            eye = (robot_x - forward_x * 4.8, robot_z - forward_z * 4.8, 3.05)
            target = (robot_x * 0.3 + 0.2, robot_z + forward_z * 2.6, 0.65)
        self.base.camera.setPos(*eye)
        self.base.camera.lookAt(self._Point3(*target))

        self.base.graphicsEngine.renderFrame()
        self.base.graphicsEngine.renderFrame()
        texture = self.base.win.getScreenshot()
        if texture is None:
            raise RuntimeError("Panda3D could not capture the offscreen frame")
        image = np.frombuffer(bytes(texture.getRamImageAs("RGB")), dtype=np.uint8)
        expected = self.height * self.width * 3
        if image.size != expected:
            raise RuntimeError(f"Panda3D returned an unexpected framebuffer size: {image.size} != {expected}")
        return image.reshape((self.height, self.width, 3))[::-1].copy()

    def close(self) -> None:
        self.base.destroy()


def render_warehouse_frame(**kwargs: Any) -> np.ndarray:
    """Render a warehouse frame with one reusable offscreen Panda3D context."""
    global _renderer
    if _renderer is None:
        _renderer = PandaWarehouseRenderer(kwargs["width"], kwargs["height"])
    return _renderer.render(**kwargs)


def close_warehouse_renderer() -> None:
    """Release the process-wide offscreen graphics context, if it was opened."""
    global _renderer
    if _renderer is not None:
        _renderer.close()
        _renderer = None
