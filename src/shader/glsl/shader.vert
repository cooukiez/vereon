#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 fs_pos;
layout(location = 1) out vec2 fs_uv;

void main() {
    fs_pos = vec4(in_pos, 1.0);
    fs_uv = in_uv;
    gl_Position = fs_pos;
}