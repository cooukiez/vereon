struct Uniform {
    view_proj: mat4x4<f32>,
    time: u32,
};
@group(0) @binding(0)
var<uniform> unif_buf: Uniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;
    out.clip_position = vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.uv * sin(f32(unif_buf.time) / 1000.0), vec2<f32>(1.0));
}