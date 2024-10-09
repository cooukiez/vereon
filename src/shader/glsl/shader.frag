#version 450

layout(location = 0) in vec4 fs_pos;
layout(location = 1) in vec2 fs_uv;

layout(location = 0) out vec4 out_col;

layout (binding = 0) uniform UBO {
    mat4 view_proj;

    vec4 cam_pos;
    vec4 cam_dir;
    vec4 cam_plane_u;
    vec4 cam_plane_v;

    vec2 res;
    vec2 mouse;

    uint time;
} ubo;

layout (binding = 1) buffer SVO {
    uint nodes[];
} svo;

struct Ray {
    vec3 o;
    vec3 d;
};

void main() {
    vec3 col = vec3(0.5) + 0.5 * cos(vec3(ubo.time) / 1000.0 + fs_uv.xyx + vec3(0,2,4));
    out_col = vec4(col, 1.0);
}