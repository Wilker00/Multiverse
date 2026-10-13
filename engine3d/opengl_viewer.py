"""Custom OpenGL 3.3 scene viewer with Dear ImGui inspection panels.

Simulation publishes snapshots independently; this window only consumes and
draws the latest scene state.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from engine3d.viewer_scene import SceneFrame, SceneObject, SceneSnapshotChannel, scene_from_verse


_VERTEX_SHADER = """#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uLightSpace;
out vec3 vWorldPosition;
out vec3 vNormal;
out vec4 vLightPosition;
void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    vLightPosition = uLightSpace * world;
    gl_Position = uProjection * uView * world;
}
"""
_FRAGMENT_SHADER = """#version 330 core
in vec3 vWorldPosition;
in vec3 vNormal;
in vec4 vLightPosition;
uniform vec3 uTint;
uniform vec3 uMaterialColor;
uniform vec3 uLightDirection;
uniform vec3 uCameraPosition;
uniform float uAmbient;
uniform float uRoughness;
uniform sampler2D uShadowMap;
out vec4 fragmentColor;
float shadow_amount(vec3 normal, vec3 lightDir) {
    vec3 projected = vLightPosition.xyz / vLightPosition.w;
    projected = projected * 0.5 + 0.5;
    if (projected.z > 1.0 || any(lessThan(projected.xy, vec2(0.0))) ||
        any(greaterThan(projected.xy, vec2(1.0)))) return 0.0;
    float bias = max(0.0007 * (1.0 - dot(normal, lightDir)), 0.00018);
    vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));
    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float closest = texture(uShadowMap, projected.xy + vec2(x, y) * texel).r;
            shadow += projected.z - bias > closest ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}
void main() {
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(-uLightDirection);
    float diffuse = max(dot(n, lightDir), 0.0);
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    vec3 halfDir = normalize(lightDir + viewDir);
    float exponent = mix(96.0, 5.0, clamp(uRoughness, 0.0, 1.0));
    float specular = pow(max(dot(n, halfDir), 0.0), exponent) * (1.0 - uRoughness) * 0.22;
    float shadow = shadow_amount(n, lightDir);
    vec3 lit = uTint * uMaterialColor * (uAmbient + diffuse * (1.0 - shadow * 0.68)) + vec3(specular);
    fragmentColor = vec4(pow(max(lit, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
"""
_DEPTH_VERTEX_SHADER = """#version 330 core
layout(location=0) in vec3 aPosition;
uniform mat4 uModel;
uniform mat4 uLightSpace;
void main() { gl_Position = uLightSpace * uModel * vec4(aPosition, 1.0); }
"""
_DEPTH_FRAGMENT_SHADER = """#version 330 core
void main() {}
"""


def _perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = f / max(aspect, 1e-6)
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _orthographic(left: float, right: float, bottom: float, top: float,
                  near: float, far: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 2.0 / (right - left)
    matrix[1, 1] = 2.0 / (top - bottom)
    matrix[2, 2] = -2.0 / (far - near)
    matrix[0, 3] = -(right + left) / (right - left)
    matrix[1, 3] = -(top + bottom) / (top - bottom)
    matrix[2, 3] = -(far + near) / (far - near)
    return matrix


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, forward)
    result = np.eye(4, dtype=np.float64)
    result[0, :3], result[1, :3], result[2, :3] = right, up, -forward
    result[0, 3] = -float(np.dot(right, eye))
    result[1, 3] = -float(np.dot(up, eye))
    result[2, 3] = float(np.dot(forward, eye))
    return result


def _model_matrix(position: Sequence[float], scale: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    px, py, pz = map(float, position)
    sx, sy, sz = map(float, scale)
    rx, ry, rz = map(float, rotation)
    cx, sxr = math.cos(rx), math.sin(rx)
    cy, syr = math.cos(ry), math.sin(ry)
    cz, szr = math.cos(rz), math.sin(rz)
    rxm = np.array([[1,0,0,0],[0,cx,-sxr,0],[0,sxr,cx,0],[0,0,0,1]], dtype=float)
    rym = np.array([[cy,0,syr,0],[0,1,0,0],[-syr,0,cy,0],[0,0,0,1]], dtype=float)
    rzm = np.array([[cz,-szr,0,0],[szr,cz,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
    translate = np.eye(4)
    translate[:3, 3] = (px, py, pz)
    return translate @ rym @ rxm @ rzm @ np.diag((sx, sy, sz, 1.0))


def _align_y(direction: np.ndarray) -> np.ndarray:
    target = direction / max(np.linalg.norm(direction), 1e-12)
    source = np.array([0.0, 1.0, 0.0])
    axis = np.cross(source, target)
    cosine = float(np.dot(source, target))
    sine = float(np.linalg.norm(axis))
    if sine < 1e-10:
        return np.eye(3) if cosine >= 0 else np.diag((1.0, -1.0, -1.0))
    skew = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _mesh_data(shape: str) -> Tuple[np.ndarray, np.ndarray]:
    vertices, indices = [], []
    if shape == "sphere":
        rings, sectors = 16, 24
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for sector in range(sectors + 1):
                theta = 2 * math.pi * sector / sectors
                normal = (math.sin(phi)*math.cos(theta), math.cos(phi), math.sin(phi)*math.sin(theta))
                vertices.append([*(0.5*v for v in normal), *normal])
        for ring in range(rings):
            for sector in range(sectors):
                a = ring*(sectors+1)+sector
                b = a+sectors+1
                indices.extend((a,b,a+1,a+1,b,b+1))
    elif shape == "cylinder":
        sectors = 24
        for end in (-0.5, 0.5):
            for sector in range(sectors):
                theta = 2*math.pi*sector/sectors
                nx, nz = math.cos(theta), math.sin(theta)
                vertices.append([0.5*nx,end,0.5*nz,nx,0.0,nz])
        for sector in range(sectors):
            nxt = (sector+1) % sectors
            indices.extend((sector,nxt,sectors+nxt,sector,sectors+nxt,sectors+sector))
    else:
        faces = [
            (((-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)),(0,0,1)),
            (((1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1)),(0,0,-1)),
            (((-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1)),(-1,0,0)),
            (((1,-1,1),(1,-1,-1),(1,1,-1),(1,1,1)),(1,0,0)),
            (((-1,1,1),(1,1,1),(1,1,-1),(-1,1,-1)),(0,1,0)),
            (((-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1)),(0,-1,0)),
        ]
        for corners, normal in faces:
            base = len(vertices)
            for corner in corners:
                vertices.append([*(0.5*v for v in corner), *normal])
            indices.extend((base,base+1,base+2,base,base+2,base+3))
    return np.asarray(vertices, np.float32), np.asarray(indices, np.uint32)


class OpenGLSceneViewer:
    """Interactive custom viewport with a scene tree and property inspectors."""

    def __init__(self, *, width: int = 1440, height: int = 900, title: str = "Multiverse 3D Viewer"):
        try:
            import glfw
            from OpenGL import GL
            from imgui_bundle import imgui
            from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements_viewer.txt to enable the GLFW/OpenGL/Dear ImGui viewer."
            ) from exc
        self.glfw, self.gl, self.imgui, self._backend_type = glfw, GL, imgui, GlfwRenderer
        self.width, self.height, self.title = int(width), int(height), title
        self.window = None
        self.backend = None
        self.program = 0
        self.depth_program = 0
        self.shadow_framebuffer = 0
        self.shadow_texture = 0
        self.shadow_size = 2048
        self.uniforms: Dict[str, int] = {}
        self.meshes: Dict[str, Tuple[int, int, int, int, Tuple[float, float, float], float]] = {}
        self.selected: Optional[str] = None
        self.tint_overrides: Dict[str, Tuple[float, float, float]] = {}
        self.roughness: Dict[str, float] = {}
        self.position_overrides: Dict[str, Tuple[float, float, float]] = {}
        self.light = np.array([-0.45, -1.0, -0.35], dtype=float)
        self.ambient = 0.28
        self.azimuth, self.elevation, self.distance = 0.78, 0.58, 12.0
        self.target = np.array([0.0, 0.5, 0.0], dtype=float)
        self.last_cursor = None
        self.scroll_delta = 0.0
        self.last_time = time.perf_counter()
        self.fps = 0.0
        self.ui_placed = False
        self.view_initialized = False

    def _init(self) -> None:
        glfw, gl, imgui = self.glfw, self.gl, self.imgui
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize a graphics context")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create an OpenGL 3.3 window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        imgui.create_context()
        imgui.style_colors_dark()
        self.backend = self._backend_type(self.window)
        self._create_program()
        self._create_shadow_target()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glClearColor(0.055, 0.07, 0.09, 1.0)
        self.old_scroll_callback = glfw.set_scroll_callback(self.window, self._on_scroll)

    def _link_program(self, vertex_source: str, fragment_source: str) -> int:
        gl = self.gl
        shaders = []
        for kind, source in ((gl.GL_VERTEX_SHADER, vertex_source), (gl.GL_FRAGMENT_SHADER, fragment_source)):
            shader = gl.glCreateShader(kind)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                raise RuntimeError(gl.glGetShaderInfoLog(shader).decode("utf-8", "replace"))
            shaders.append(shader)
        program = gl.glCreateProgram()
        for shader in shaders:
            gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(program).decode("utf-8", "replace"))
        for shader in shaders:
            gl.glDeleteShader(shader)
        return int(program)

    def _create_program(self) -> None:
        gl = self.gl
        self.program = self._link_program(_VERTEX_SHADER, _FRAGMENT_SHADER)
        self.depth_program = self._link_program(_DEPTH_VERTEX_SHADER, _DEPTH_FRAGMENT_SHADER)
        names = ("uModel", "uView", "uProjection", "uLightSpace", "uLightDirection",
                 "uCameraPosition", "uAmbient", "uRoughness", "uTint", "uMaterialColor", "uShadowMap")
        self.uniforms = {name: int(gl.glGetUniformLocation(self.program, name)) for name in names}

    def _create_shadow_target(self) -> None:
        gl = self.gl
        self.shadow_framebuffer = int(gl.glGenFramebuffers(1))
        self.shadow_texture = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.shadow_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24,
                        self.shadow_size, self.shadow_size, 0,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR,
                            np.asarray((1.0, 1.0, 1.0, 1.0), dtype=np.float32))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.shadow_framebuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                  gl.GL_TEXTURE_2D, self.shadow_texture, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"OpenGL shadow framebuffer is incomplete (0x{int(status):x})")

    def _on_scroll(self, window, x_offset, y_offset):
        old = getattr(self, "old_scroll_callback", None)
        if old is not None:
            try:
                old(window, x_offset, y_offset)
            except TypeError:
                pass
        if not self.imgui.get_io().want_capture_mouse:
            self.scroll_delta += float(y_offset)

    def _camera(self):
        horizontal = math.cos(self.elevation) * self.distance
        eye = self.target + np.array([
            horizontal*math.sin(self.azimuth),
            math.sin(self.elevation)*self.distance,
            horizontal*math.cos(self.azimuth),
        ])
        return eye, _look_at(eye, self.target)

    def _update_camera(self) -> None:
        glfw = self.glfw
        cursor = tuple(map(float, glfw.get_cursor_pos(self.window)))
        if self.last_cursor is not None and not self.imgui.get_io().want_capture_mouse:
            dx, dy = cursor[0]-self.last_cursor[0], cursor[1]-self.last_cursor[1]
            if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                self.azimuth += dx*0.008
                self.elevation = float(np.clip(self.elevation-dy*0.008, -1.48, 1.48))
            pan = any(glfw.get_mouse_button(self.window, b) == glfw.PRESS for b in
                      (glfw.MOUSE_BUTTON_MIDDLE, glfw.MOUSE_BUTTON_RIGHT))
            if pan:
                eye, _ = self._camera()
                forward = (self.target-eye)/max(np.linalg.norm(self.target-eye), 1e-12)
                right = np.cross(forward, np.array([0.0,1.0,0.0]))
                right /= max(np.linalg.norm(right), 1e-12)
                up = np.cross(right, forward)
                self.target += (-right*dx+up*dy)*(self.distance*0.0018)
        self.last_cursor = cursor
        if not self.imgui.get_io().want_capture_mouse and self.scroll_delta:
            self.distance = float(np.clip(self.distance*math.exp(-self.scroll_delta*0.12), 1.0, 80.0))
        self.scroll_delta = 0.0

    def _upload_mesh(self, vertices: np.ndarray, indices: np.ndarray,
                     material=(1.0,1.0,1.0), roughness=0.72):
        gl = self.gl
        vao, vbo, ebo = int(gl.glGenVertexArrays(1)), int(gl.glGenBuffers(1)), int(gl.glGenBuffers(1))
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        stride = 24
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0,3,gl.GL_FLOAT,gl.GL_FALSE,stride,ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1,3,gl.GL_FLOAT,gl.GL_FALSE,stride,ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        return vao,vbo,ebo,int(indices.size),tuple(float(v) for v in material),float(roughness)

    def _gltf_mesh(self, path: str):
        try:
            import trimesh
        except ImportError as exc:
            raise RuntimeError("glTF loading requires trimesh from requirements_viewer.txt") from exc
        scene = trimesh.load(path, force="scene", process=False)
        vertices, indices, offset = [], [], 0
        material_color = (1.0,1.0,1.0)
        material_roughness = 0.72
        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node]
            mesh = scene.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
            positions = np.asarray(mesh.vertices, dtype=np.float32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            vertices.append(np.concatenate((positions,normals),axis=1))
            faces = np.asarray(mesh.faces,dtype=np.uint32).reshape(-1,3)
            indices.append((faces+offset).reshape(-1))
            offset += len(positions)
            material = getattr(getattr(mesh,"visual",None),"material",None)
            factor = getattr(material,"baseColorFactor",None)
            if factor is not None:
                factor = np.asarray(factor,dtype=float).reshape(-1)[:3]
                if factor.max(initial=0.0)>1.0:
                    factor /= 255.0
                material_color = tuple(float(v) for v in factor)
            pbr = getattr(material,"pbrMetallicRoughness",None)
            if pbr is not None:
                material_roughness = float(getattr(pbr,"roughnessFactor",material_roughness))
            else:
                material_roughness = float(getattr(material,"roughnessFactor",material_roughness))
        if not vertices:
            raise ValueError(f"glTF has no mesh: {path}")
        return np.concatenate(vertices),np.concatenate(indices),material_color,material_roughness

    def _get_mesh(self,obj: SceneObject):
        key = obj.mesh_path or "primitive:"+obj.shape
        if key not in self.meshes:
            if obj.mesh_path:
                vertices,indices,material,roughness = self._gltf_mesh(obj.mesh_path)
            else:
                vertices,indices = _mesh_data(obj.shape)
                material = (1.0,1.0,1.0)
                roughness = 0.72
            self.meshes[key]=self._upload_mesh(vertices,indices,material,roughness)
        return self.meshes[key]

    def _object_model(self,obj: SceneObject) -> np.ndarray:
        position=np.asarray(self.position_overrides.get(obj.name,obj.position),dtype=float)
        if obj.shape=="line" and obj.end_position is not None:
            start=np.asarray(obj.position,dtype=float)
            end=np.asarray(obj.end_position,dtype=float)
            delta=end-start
            length=max(float(np.linalg.norm(delta)),1e-6)
            model=np.eye(4)
            radius=max(0.003,float(obj.line_width)*0.5)
            model[:3,:3]=_align_y(delta) @ np.diag((radius,length,radius))
            model[:3,3]=(start+end)*0.5
            return model
        return _model_matrix(position,obj.scale,obj.rotation)

    def _object_mesh(self,obj: SceneObject):
        if obj.shape=="line" and obj.end_position is not None:
            mesh=self.meshes.get("primitive:cylinder")
            if mesh is None:
                v,i=_mesh_data("cylinder")
                mesh=self._upload_mesh(v,i)
                self.meshes["primitive:cylinder"]=mesh
            return mesh
        return self._get_mesh(obj)

    def _draw_shadow_map(self,frame: SceneFrame,light_space: np.ndarray):
        gl=self.gl
        gl.glViewport(0,0,self.shadow_size,self.shadow_size)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,self.shadow_framebuffer)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.depth_program)
        light_loc=gl.glGetUniformLocation(self.depth_program,"uLightSpace")
        model_loc=gl.glGetUniformLocation(self.depth_program,"uModel")
        gl.glUniformMatrix4fv(light_loc,1,gl.GL_TRUE,np.asarray(light_space,dtype=np.float32))
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(2.0,4.0)
        for obj in frame.objects:
            if not obj.visible:
                continue
            model=self._object_model(obj)
            vao,_,_,count,_,_=self._object_mesh(obj)
            gl.glUniformMatrix4fv(model_loc,1,gl.GL_TRUE,np.asarray(model,dtype=np.float32))
            gl.glBindVertexArray(vao)
            gl.glDrawElements(gl.GL_TRIANGLES,count,gl.GL_UNSIGNED_INT,None)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,0)

    def _draw_obj(self,obj: SceneObject,eye: np.ndarray):
        if not obj.visible:
            return
        gl=self.gl
        model=self._object_model(obj)
        mesh=self._object_mesh(obj)
        vao,_,_,count,material,asset_roughness=mesh
        tint=self.tint_overrides.get(obj.name,obj.color)
        rough=float(self.roughness.get(obj.name,asset_roughness))
        u=self.uniforms
        gl.glUniformMatrix4fv(u["uModel"],1,gl.GL_TRUE,np.asarray(model,dtype=np.float32))
        gl.glUniform3f(u["uTint"],*map(float,tint))
        gl.glUniform3f(u["uMaterialColor"],*material)
        gl.glUniform1f(u["uRoughness"],rough)
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES,count,gl.GL_UNSIGNED_INT,None)
        gl.glBindVertexArray(0)

    def _draw_world(self,frame: SceneFrame):
        gl=self.gl
        if not self.view_initialized and frame.objects:
            lows, highs = [], []
            for obj in frame.objects:
                center = np.asarray(obj.position, dtype=float)
                half = np.asarray(obj.scale, dtype=float) * 0.5
                lows.append(center-half)
                highs.append(center+half)
                if obj.end_position is not None:
                    endpoint = np.asarray(obj.end_position, dtype=float)
                    lows.append(endpoint)
                    highs.append(endpoint)
            low, high = np.min(lows, axis=0), np.max(highs, axis=0)
            self.target = (low+high)*0.5
            self.distance = float(np.clip(np.max(high-low)*1.45, 2.0, 80.0))
            self.view_initialized = True
        fw,fh=self.glfw.get_framebuffer_size(self.window)
        self.width,self.height=max(1,fw),max(1,fh)
        eye,view=self._camera()
        projection=_perspective(math.radians(52),self.width/self.height,0.03,250)
        light_direction=-self.light/max(float(np.linalg.norm(self.light)),1e-12)
        light_eye=self.target+light_direction*18.0
        light_view=_look_at(light_eye,self.target)
        light_projection=_orthographic(-9.0,9.0,-9.0,9.0,0.1,38.0)
        light_space=light_projection@light_view
        self._draw_shadow_map(frame,light_space)
        gl.glViewport(0,0,self.width,self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.program)
        u=self.uniforms
        gl.glUniformMatrix4fv(u["uView"],1,gl.GL_TRUE,np.asarray(view,dtype=np.float32))
        gl.glUniformMatrix4fv(u["uProjection"],1,gl.GL_TRUE,np.asarray(projection,dtype=np.float32))
        gl.glUniformMatrix4fv(u["uLightSpace"],1,gl.GL_TRUE,np.asarray(light_space,dtype=np.float32))
        gl.glUniform3f(u["uLightDirection"],*map(float,self.light))
        gl.glUniform3f(u["uCameraPosition"],*map(float,eye))
        gl.glUniform1f(u["uAmbient"],float(self.ambient))
        gl.glUniform1i(u["uShadowMap"],0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D,self.shadow_texture)
        for obj in frame.objects:
            self._draw_obj(obj,eye)
        gl.glBindTexture(gl.GL_TEXTURE_2D,0)
        gl.glUseProgram(0)

    def _panels(self,frame: SceneFrame):
        imgui=self.imgui
        if self.selected is None:
            self.selected=frame.selected_name
        listed_objects=[obj for obj in frame.objects if obj.inspectable]
        if not self.ui_placed:
            imgui.set_next_window_pos((12, 38))
            imgui.set_next_window_size((218, 330))
        if imgui.begin("Scene")[0]:
            imgui.text(f"{frame.title}  |  {len(listed_objects)} objects")
            imgui.separator()
            for obj in listed_objects:
                clicked,_=imgui.selectable(obj.name,obj.name==self.selected)
                if clicked:
                    self.selected=obj.name
        imgui.end()
        if not self.ui_placed:
            imgui.set_next_window_pos((max(242, self.width-292), 38))
            imgui.set_next_window_size((276, 250))
        if imgui.begin("Object")[0]:
            obj=next((item for item in frame.objects if item.name==self.selected),None)
            if obj is None:
                imgui.text("Select an object in the scene list.")
            else:
                imgui.text(obj.name)
                imgui.text("Type: "+(Path(obj.mesh_path).name if obj.mesh_path else obj.shape))
                if obj.physics_body_name:
                    imgui.text("Physics body: "+obj.physics_body_name)
                imgui.text("Position: "+", ".join(f"{v:.2f}" for v in obj.position))
                imgui.text("Size: "+", ".join(f"{v:.2f}" for v in obj.scale))
                color=self.tint_overrides.get(obj.name,obj.color)
                tint=list(color)
                for i,channel in enumerate(("R","G","B")):
                    changed,value=imgui.slider_float(channel,float(color[i]),0.0,1.0)
                    if changed:
                        tint[i]=float(value)
                self.tint_overrides[obj.name]=tuple(tint)
                changed,value=imgui.slider_float("Surface",float(self.roughness.get(obj.name,0.72)),0.04,1.0)
                if changed:
                    self.roughness[obj.name]=float(value)
        imgui.end()
        if not self.ui_placed:
            imgui.set_next_window_pos((max(242, self.width-292), 298))
            imgui.set_next_window_size((276, 174))
        if imgui.begin("Status")[0]:
            for key,value in frame.telemetry.items():
                if isinstance(value,(tuple,list)):
                    value=", ".join(f"{v:.3f}" if isinstance(v,(float,int)) else str(v) for v in value)
                elif isinstance(value,float):
                    value=f"{value:.4f}"
                imgui.text(f"{key}: {value}")
            imgui.text(f"Draw rate: {self.fps:.0f} fps")
        imgui.end()
        if not self.ui_placed:
            imgui.set_next_window_pos((max(242, self.width-292), 482))
            imgui.set_next_window_size((276, 210))
        if imgui.begin("View")[0]:
            imgui.text("Drag to orbit  |  Right drag to pan")
            imgui.text("Scroll to zoom")
            imgui.separator()
            changed,value=imgui.slider_float("Ambient",float(self.ambient),0.05,0.8)
            if changed:
                self.ambient=float(value)
            for i,label in enumerate(("Light X","Light Y","Light Z")):
                changed,value=imgui.slider_float(label,float(self.light[i]),-1.0,1.0)
                if changed:
                    self.light[i]=float(value)
        imgui.end()
        self.ui_placed=True

    def render_frame(self,frame: SceneFrame) -> bool:
        if self.window is None:
            self._init()
        if self.glfw.window_should_close(self.window):
            return False
        self.glfw.poll_events()
        self._update_camera()
        now=time.perf_counter()
        self.fps=1.0/max(now-self.last_time,1e-6)
        self.last_time=now
        self._draw_world(frame)
        self.backend.process_inputs()
        self.imgui.new_frame()
        self._panels(frame)
        self.imgui.render()
        self.backend.render(self.imgui.get_draw_data())
        self.glfw.swap_buffers(self.window)
        return True

    def close(self):
        if self.window is None:
            return
        gl,glfw=self.gl,self.glfw
        try:
            self.backend.shutdown()
            self.imgui.destroy_context()
            for vao,vbo,ebo,_,_,_ in self.meshes.values():
                gl.glDeleteBuffers(1,[vbo])
                gl.glDeleteBuffers(1,[ebo])
                gl.glDeleteVertexArrays(1,[vao])
            if self.program:
                gl.glDeleteProgram(self.program)
            if self.depth_program:
                gl.glDeleteProgram(self.depth_program)
            if self.shadow_framebuffer:
                gl.glDeleteFramebuffers(1,[self.shadow_framebuffer])
            if self.shadow_texture:
                gl.glDeleteTextures(1,[self.shadow_texture])
            glfw.destroy_window(self.window)
        finally:
            glfw.terminate()
            self.window=None


def run_viewer(frame_provider: Callable[[],SceneFrame], *, width=1440,height=900,title="Multiverse 3D Viewer"):
    """Draw snapshots at the viewer's rate; physics timing belongs to the caller."""
    viewer=OpenGLSceneViewer(width=width,height=height,title=title)
    try:
        while True:
            if not viewer.render_frame(frame_provider()):
                break
    finally:
        viewer.close()


def view_verse(verse: Any, *, tick: Optional[Callable[[Any],None]]=None):
    """Convenience wrapper; optional tick is called independently by the UI loop."""
    channel=SceneSnapshotChannel()
    def publish():
        if tick is not None:
            tick(verse)
        frame=scene_from_verse(verse)
        channel.publish(frame)
        return channel.read_latest()[1] or frame
    run_viewer(publish,title=f"Multiverse | {verse.spec.verse_name}")


__all__=["OpenGLSceneViewer","run_viewer","view_verse"]

