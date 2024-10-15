#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 fs_pos;
layout(location = 1) out vec2 fs_uv;

layout (binding = 0) uniform UBO {
    mat4 view_proj;

    vec4 cam_pos;
    vec4 cam_dir;
    vec4 cam_plane_u;
    vec4 cam_plane_v;

    uvec2 res;
    vec2 mouse;

    uint time;
} ubo;

void main() {
    fs_pos = ubo.view_proj * vec4(in_pos, 1.0);
    fs_uv = in_uv;
    gl_Position = fs_pos;
}