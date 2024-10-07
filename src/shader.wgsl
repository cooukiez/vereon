struct Uniform {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    cam_dir: vec4<f32>,
    cam_plane_u: vec4<f32>,
    cam_plane_v: vec4<f32>,
    res: vec2<u32>,
    mouse: vec2<f32>,
    time: u32,
};

@group(0) @binding(0)
var<uniform> ubo: Uniform;
@group(0) @binding(1)
var<storage, read> svo: array<u32>;

const EPS: f32 = 3.5e-15;
const STACK_SIZE: u32 = 23;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Ray {
    o: vec3<f32>,
    d: vec3<f32>,
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
    // let col = vec3<f32>(0.5) + 0.5 * cos(vec3<f32>(ubo.time) / 1000.0 + in.uv.xyx + vec3(0,2,4));

    var screen_pos: vec2<f32> = (in.uv * 2.0 - vec2<f32>(1.0)) * vec2<f32>(f32(ubo.res.x) / f32(ubo.res.y), 1.0);
    var dir: vec3<f32> = ubo.cam_dir.xyz + screen_pos.x * ubo.cam_plane_u.xyz + screen_pos.y * ubo.cam_plane_v.xyz;

    var r: Ray = Ray(
        ubo.cam_pos.xyz,
        dir,
    );

    var pos: vec3<f32>;
    var col: vec3<f32>;
    var norm: vec3<f32>;
    var iter: u32;

    var result: bool = raymarch_leaf(
        r,
        &pos,
        &col,
        &norm,
        &iter,
    );

    return vec4<f32>(col, 1.0);
}

/*
*
*   code is from theses repositories:
*   https://code.google.com/archive/p/efficient-sparse-voxel-octrees/
*   https://github.com/AdamYuan/SparseVoxelOctree/blob/master/shader/octree.glsl
*
*/
fn raymarch_leaf(
    r: Ray,
    p_pos: ptr<function, vec3<f32>>,
    p_col: ptr<function, vec3<f32>>,
    p_norm: ptr<function, vec3<f32>>,
    p_iter: ptr<function, u32>
) -> bool {
    var iter: u32 = 0;
    var stack: array<u32, STACK_SIZE>;

    var o: vec3<f32> = r.o;
    var d: vec3<f32> = r.d;
    var d_abs: vec3<f32> = abs(d);

    // get rid of small direction components, to avoid division by zero
    d.x = select(select(-EPS, EPS, d.x >= 0), d.x, d_abs.x > EPS);
    d.y = select(select(-EPS, EPS, d.y >= 0), d.y, d_abs.y > EPS);
    d.z = select(select(-EPS, EPS, d.z >= 0), d.z, d_abs.z > EPS);

    // precompute coefficients of tx(x), ty(y), tz(z)
    // octree is assumed to reside at coordinates [1, 2]
    var t_coef: vec3<f32> = -1.0 / d_abs;
    var t_bias: vec3<f32> = t_coef * o;

    // select octant mask to mirror the coordinate system,
    // so that ray direction is negative along axis
    var oct_mask: u32 = 0u;
    if (d.x > 0.0) {
        oct_mask = oct_mask ^ 1u;
        t_bias.x = 3.0 * t_coef.x - t_bias.x;
    }
    if (d.y > 0.0) {
        oct_mask = oct_mask ^ 2u;
        t_bias.y = 3.0 * t_coef.y - t_bias.y;
    }
    if (d.z > 0.0) {
        oct_mask = oct_mask ^ 4u;
        t_bias.z = 3.0 * t_coef.z - t_bias.z;
    }

    // intialize active span of t-values
    var t_min: f32 = max(max(2.0 * t_coef.x - t_bias.x, 2.0 * t_coef.y - t_bias.y), 2.0 * t_coef.z - t_bias.z);
    var t_max: f32 = min(min(t_coef.x - t_bias.x, t_coef.y - t_bias.y), t_coef.z - t_bias.z);
    t_min = max(t_min, 0.0);
    var h: f32 = t_max;

    // initialize current voxel to first child of root
    var parent: u32 = 0u; // first child index
    var cur: u32 = 0u; // the current selected node
    var pos: vec3<f32> = vec3<f32>(1.0);
    var idx: u32 = 0u; // child index

    if (1.5 * t_coef.x - t_bias.x > t_min) {
        idx = idx ^ 1u;
        pos.x = 1.5;
    }
    if (1.5 * t_coef.y - t_bias.y > t_min) {
        idx = idx ^ 2u;
        pos.y = 1.5;
    }
    if (1.5 * t_coef.z - t_bias.z > t_min) {
        idx = idx ^ 4u;
        pos.z = 1.5;
    }

    var scale: u32 = STACK_SIZE - 1;
    var scale_exp2: f32 = 0.5; // exp2( scale - STACK_SIZE )

    // traverse voxels along the ray
    // as long as the current voxel stays within the octree
    while (scale < STACK_SIZE) {
        iter++;

        // fetch child descriptor unless it is already valid
        if (cur == 0u) {
            // READ
            cur = svo[parent + (idx ^ oct_mask)];
        }

        // determine maximum t-value of the cube by
        // evaluating tx(), ty(), tz() at its corner
        var t_corner: vec3<f32> = pos * t_coef - t_bias;
        var tc_max: f32 = min(min(t_corner.x, t_corner.y), t_corner.z);

        // process voxel if the corresponding bit in
        // if valid mask is set and the active t-span is non-empty
        if ((cur & 0x80000000u) != 0 && t_min <= t_max) {
            // INTERSECT
            // intersect active t-span with the cube and evaluate
            // tx(), ty(), tz() at the center of the voxel
            var half_scale_exp2: f32 = scale_exp2 * 0.5;
            var t_center: vec3<f32> = half_scale_exp2 * t_coef + t_corner;

            // leaf node
            if ((cur & 0x40000000u) != 0) {
                break;
            }

            // push to stack
            if (tc_max < h) {
                stack[scale] = parent;
            }
            h = tc_max;

            parent = cur & 0x3fffffffu;

            // select child voxel that the ray enters first
            idx = 0u;
            scale--;
            scale_exp2 = half_scale_exp2;
            if (t_center.x > t_min) {
                idx = idx ^ 1u;
                pos.x += scale_exp2;
            }
            if (t_center.y > t_min) {
                idx = idx ^ 2u;
                pos.y += scale_exp2;
            }
            if (t_center.z > t_min) {
                idx = idx ^ 4u;
                pos.z += scale_exp2;
            }

            cur = 0u;

            continue;
        }

        // ADVANCE
        // step along the ray
        var step_mask: u32 = 0u;
        if (t_corner.x <= tc_max) {
            step_mask = step_mask ^ 1u;
            pos.x -= scale_exp2;
        }
        if (t_corner.y <= tc_max) {
            step_mask = step_mask ^ 2u;
            pos.y -= scale_exp2;
        }
        if (t_corner.z <= tc_max) {
            step_mask = step_mask ^ 4u;
            pos.z -= scale_exp2;
        }

        // update active t-span and flip bits of the child slot index
        t_min = tc_max;
        idx = idx ^ step_mask;

        if ((idx & step_mask) != 0) {
            var differing_bits: u32 = 0;

            // POP
            // find the highest differing bit between the two positions
            if ((step_mask & 1u) != 0) {
                differing_bits |= (bitcast<u32>(pos.x) ^ bitcast<u32>(pos.x + scale_exp2));
            }
            if ((step_mask & 2u) != 0) {
                differing_bits |= (bitcast<u32>(pos.y) ^ bitcast<u32>(pos.y + scale_exp2));
            }
            if ((step_mask & 4u) != 0) {
                differing_bits |= (bitcast<u32>(pos.z) ^ bitcast<u32>(pos.z + scale_exp2));
            }
            // position of the highest differing bit
            scale = firstLeadingBit(differing_bits);
            if (scale >= STACK_SIZE) {
                break;
            }

            scale_exp2 = bitcast<f32>((scale - STACK_SIZE + 127u) << 23u); // exp2f(scale - s_max)

            // restore parent voxel from the stack
            parent = stack[scale];

            // round cube position and extract child slot index
            var shx: u32 = bitcast<u32>(pos.x) >> scale;
            var shy: u32 = bitcast<u32>(pos.y) >> scale;
            var shz: u32 = bitcast<u32>(pos.z) >> scale;
            pos.x = bitcast<f32>(shx << scale);
            pos.y = bitcast<f32>(shy << scale);
            pos.z = bitcast<f32>(shz << scale);
            idx = (shx & 1u) | ((shy & 1u) << 1u) | ((shz & 1u) << 2u);

            // prevent same parent from being stored
            // again and invalidate cached child descriptor
            h = 0.0;
            cur = 0u;
        }
    }

    var t_corner: vec3<f32> = t_coef * (pos + scale_exp2) - t_bias;

    var norm: vec3<f32> = select(vec3<f32>(-1, 0, 0),
        select(vec3<f32>(0, -1, 0),
        vec3<f32>(0, 0, -1),
        t_corner.y > t_corner.z),
        t_corner.x > t_corner.y && t_corner.x > t_corner.z);

    // undo mirroring of the coordinate system
    if ((oct_mask & 1u) == 0u) {
        norm.x = -norm.x;
    }
    if ((oct_mask & 2u) == 0u) {
        norm.y = -norm.y;
    }
    if ((oct_mask & 4u) == 0u) {
        norm.z = -norm.z;
    }

    // undo mirroring of the coordinate system
    if ((oct_mask & 1u) != 0u) {
        pos.x = 3.0 - scale_exp2 - pos.x;
    }
    if ((oct_mask & 2u) != 0u) {
        pos.y = 3.0 - scale_exp2 - pos.y;
    }
    if ((oct_mask & 4u) != 0u) {
        pos.z = 3.0 - scale_exp2 - pos.z;
    }

    // output results
    *p_pos = clamp(o + t_min * d, pos, pos + scale_exp2);
    if (norm.x != 0.0) {
        (*p_pos).x = select(pos.x - EPS, pos.x + scale_exp2 + EPS * 2.0, norm.x > 0.0);
    }
    if (norm.y != 0.0) {
        (*p_pos).y = select(pos.y - EPS, pos.y + scale_exp2 + EPS * 2.0, norm.y > 0.0);
    }
    if (norm.z != 0.0) {
        (*p_pos).z = select(pos.z - EPS, pos.z + scale_exp2 + EPS * 2.0, norm.z > 0.0);
    }
    *p_norm = norm;
    *p_col = unpack4x8unorm(cur).xyz;
    *p_iter = iter;

    return scale < STACK_SIZE && t_min <= t_max;
}