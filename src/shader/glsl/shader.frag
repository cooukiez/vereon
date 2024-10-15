#version 450

layout (location = 0) in vec4 fs_pos;
layout (location = 1) in vec2 fs_uv;

layout (location = 0) out vec4 out_col;

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

layout (binding = 1) buffer SVO {
    uint nodes[];
} svo;

struct Ray {
    vec3 o;
    vec3 d;
};

#define STACK_SIZE 23
#define EPS 1e-6
#define MAX_ITER 64
#define CHILD_OFFSET 24
uint stack[STACK_SIZE];

bool raymarch(vec3 o,
              vec3 d,
out vec3 o_pos,
out vec3 o_norm,
out uint mat_data,
out uint iter,
out vec3 pos_before_start
) {
    iter = 0;
    mat_data = 0;
    vec3 d_abs = abs(d);

    // get rid of small direction components, to avoid division by zero
    d.x = d_abs.x > EPS ? d.x : (d.x >= 0 ? EPS : -EPS);
    d.y = d_abs.y > EPS ? d.y : (d.y >= 0 ? EPS : -EPS);
    d.z = d_abs.z > EPS ? d.z : (d.z >= 0 ? EPS : -EPS);

    // precompute coefficients of tx(x), ty(y), tz(z)
    // octree is assumed to reside at coordinates [1, 2]
    vec3 t_coef = -1.0 / d_abs;
    vec3 t_bias = t_coef * (o + vec3(1.0));

    uint oct_mask = 0;
    vec3 inverse_origin = 3.0 * t_coef - t_bias;
    if (d.x > 0.0f) oct_mask ^= 1u, t_bias.x = inverse_origin.x;
    if (d.y > 0.0f) oct_mask ^= 2u, t_bias.y = inverse_origin.y;
    if (d.z > 0.0f) oct_mask ^= 4u, t_bias.z = inverse_origin.z;

    // enter at [1,1,1]
    vec3 t_entry = 2.0 * t_coef - t_bias;
    float t_min = max(max(t_entry.x, t_entry.y), t_entry.z);
    // exit at [2,2,2]
    vec3 t_exit = t_coef - t_bias;
    float t_max = min(min(t_exit.x, t_exit.y), t_exit.z);

    float h = t_max;
    t_min = max(t_min, 0.0f);

    uint parent = svo.nodes[0];
    uint cur = 0;
    uint idx = 0;

    vec3 pos = vec3(1.0);
    vec3 t_half = 1.5 * t_coef - t_bias;
    if (t_half.x > t_min) idx ^= 1u, pos.x = 1.5f;
    if (t_half.y > t_min) idx ^= 2u, pos.y = 1.5f;
    if (t_half.z > t_min) idx ^= 4u, pos.z = 1.5f;

    pos_before_start = pos;

    uint scale = STACK_SIZE - 1;
    float span = 0.5; // exp2( scale - STACK_SIZE )

    // traverse voxels along the ray
    // as long as the current voxel stays within the octree
    while (iter < MAX_ITER) {
        iter++;

        // fetch child descriptor unless it is already valid
        if (cur == 0)
        cur = parent;

        // determine maximum t-value of the cube by
        // evaluating tx(), ty(), tz() at its corner
        vec3 t_corner = pos * t_coef - t_bias;
        float tc_max = min(min(t_corner.x, t_corner.y), t_corner.z);

        // process child if bit in child mask is set
        uint child_bit = 1 << (CHILD_OFFSET + (idx ^ oct_mask));
        if ((cur & child_bit) != 0 && t_min <= t_max) {
            // Todo: terminate if the voxel is small enough

            // INTERSECT
            // intersect active t-span with the cube and evaluate
            // tx(), ty(), tz() at the center of the voxel
            float tv_max = min(t_max, tc_max);
            float half_span = span * 0.5;
            vec3 t_center = half_span * t_coef + t_corner;

            // PUSH
            // write current parent to the stack
            if (tc_max < h) stack[scale] = parent;
            h = tc_max;

            // find child descriptor corresponding to the current voxel
            parent = svo.nodes[(cur & 0xFFFFFF) + (idx ^ oct_mask)];

            // check if node is a leaf
            if ((parent & 0xFF000000) == 0 && (parent & 0xFFFFFF) != 0) {
                mat_data = parent & 0xFFFFFF;
                break;
            }

            // select child voxel that the ray enters first
            idx = 0;
            scale--;
            span = half_span;
            if (t_center.x > t_min) idx ^= 1, pos.x += span;
            if (t_center.y > t_min) idx ^= 2, pos.y += span;
            if (t_center.z > t_min) idx ^= 4, pos.z += span;
            cur = 0;

            continue;
        }

        // ADVANCE
        // step along the ray to the next voxel
        uint step_mask = 0;
        if (t_corner.x <= tc_max) step_mask ^= 1, pos.x -= span;
        if (t_corner.y <= tc_max) step_mask ^= 2, pos.y -= span;
        if (t_corner.z <= tc_max) step_mask ^= 4, pos.z -= span;

        // update active t-span and flip bits of the child slot index
        t_min = tc_max;
        idx ^= step_mask;

        // proceed with pop if the bit flips disagree
        // with the ray direction
        if ((idx & step_mask) != 0) {
            // POP
            // find the highest differing bit between the two positions
            uint differing = 0; // differing bits
            if ((step_mask & 1) != 0) differing |= floatBitsToUint(pos.x) ^ floatBitsToUint(pos.x + span);
            if ((step_mask & 2) != 0) differing |= floatBitsToUint(pos.y) ^ floatBitsToUint(pos.y + span);
            if ((step_mask & 4) != 0) differing |= floatBitsToUint(pos.z) ^ floatBitsToUint(pos.z + span);
            scale = findMSB(differing);
            if (scale >= STACK_SIZE) break;
            span = uintBitsToFloat((scale - STACK_SIZE + 127u) << 23u); // exp2f(scale - s_max)

            // restore parent voxel from the stack
            parent = stack[scale];

            // round cube position and extract child slot index
            uint shx = floatBitsToUint(pos.x) >> scale;
            uint shy = floatBitsToUint(pos.y) >> scale;
            uint shz = floatBitsToUint(pos.z) >> scale;
            pos.x = uintBitsToFloat(shx << scale);
            pos.y = uintBitsToFloat(shy << scale);
            pos.z = uintBitsToFloat(shz << scale);
            idx = (shx & 1u) | ((shy & 1u) << 1u) | ((shz & 1u) << 2u);

            // prevent same parent from being stored
            // again and invalidate cached child descriptor
            h = 0.0;
            cur = 0;
        }
    }

    vec3 t_corner = t_coef * (pos + span) - t_bias;
    vec3 norm = (t_corner.x > t_corner.y && t_corner.x > t_corner.z)
    ? vec3(-1, 0, 0)
    : (t_corner.y > t_corner.z ? vec3(0, -1, 0) : vec3(0, 0, -1));
    if ((oct_mask & 1u) == 0u) norm.x = -norm.x;
    if ((oct_mask & 2u) == 0u) norm.y = -norm.y;
    if ((oct_mask & 4u) == 0u) norm.z = -norm.z;

    if ((oct_mask & 1u) != 0u) pos.x = 3.0f - span - pos.x;
    if ((oct_mask & 2u) != 0u) pos.y = 3.0f - span - pos.y;
    if ((oct_mask & 4u) != 0u) pos.z = 3.0f - span - pos.z;
    o_pos = clamp(o + t_min * d, pos, pos + span);

    if (norm.x != 0) o_pos.x = norm.x > 0 ? pos.x + span + EPS * 2 : pos.x - EPS;
    if (norm.y != 0) o_pos.y = norm.y > 0 ? pos.y + span + EPS * 2 : pos.y - EPS;
    if (norm.z != 0) o_pos.z = norm.z > 0 ? pos.z + span + EPS * 2 : pos.z - EPS;
    o_norm = norm;

    return scale < STACK_SIZE && t_min <= t_max;
}

void main() {
    // vec3 col = vec3(0.5) + 0.5 * cos(vec3(ubo.time) / 1000.0 + fs_uv.xyx + vec3(0,2,4));


    vec2 screen_pos = (fs_uv * 2.0 - 1.0) * vec2(float(ubo.res.x) / float(ubo.res.y), 1.0);
    vec3 dir = ubo.cam_dir.xyz + screen_pos.x * ubo.cam_plane_u.xyz + screen_pos.y * ubo.cam_plane_v.xyz;
    Ray ray = Ray(ubo.cam_pos.xyz, dir);

    vec3 pos, norm;
    uint mat_data, iter;
    vec3 pos_before_start;

    bool result = raymarch(ray.o, ray.d, pos, norm, mat_data, iter, pos_before_start);
    out_col = vec4(float(mat_data));
    // out_col = vec4(step(vec3(1.0), pos_before_start) * float(result), 1.0);
}
