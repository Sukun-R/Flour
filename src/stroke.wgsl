struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct SegmentInstance {
    @location(0) point_prev: vec2<f32>,
    @location(1) point_a: vec2<f32>,
    @location(2) point_b: vec2<f32>,
    @location(3) point_next: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) width: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

const MITER_LIMIT: f32 = 2.0;

fn safe_normalize(v: vec2<f32>) -> vec2<f32> {
    let len = length(v);
    if len < 0.0001 {
        return vec2(1.0, 0.0);
    }
    return v / len;
}

fn seg_normal(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = safe_normalize(b - a);
    return vec2(-d.y, d.x);
}

fn miter_vec(nor_ab: vec2<f32>, nor_other: vec2<f32>) -> vec2<f32> {
    let sum = nor_ab + nor_other;

    if length(sum) > MITER_LIMIT {
        return nor_ab;
    }
    let miter = safe_normalize(nor_ab + nor_other);
    let len = 1.0 / max(dot(miter, nor_ab), 0.0001);
    if len > MITER_LIMIT {
        return nor_ab;
    }
    return miter * len;
}

@vertex
fn vs_stroke(
    @builtin(vertex_index) vid: u32,
    inst: SegmentInstance,
) -> VertexOutput {
    let nor_ab = seg_normal(inst.point_a, inst.point_b);
    let nor_prev = seg_normal(inst.point_prev, inst.point_a);
    let nor_next = seg_normal(inst.point_b, inst.point_next);

    let miter_a = miter_vec(nor_ab, nor_prev);
    let miter_b = miter_vec(nor_ab, nor_next);

    let corners = array<vec2<f32>, 6>(
        vec2(0.0, -1.0), // A右
        vec2(0.0, 1.0), // A左
        vec2(1.0, 1.0), // B左
        vec2(0.0, -1.0), // A右
        vec2(1.0, 1.0), // B左
        vec2(1.0, -1.0), // B右
    );

    let c = corners[vid];
    let is_b_end = c.x > 0.5;

    var pos2d: vec2<f32>;
    if is_b_end {
        pos2d = inst.point_b + miter_b * inst.width * c.y;
    } else {
        pos2d = inst.point_a + miter_a * inst.width * c.y;
    }

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos2d, 0.0, 1.0);
    out.color = inst.color;
    out.uv = c;
    return out;
}

@fragment
fn fs_stroke(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = abs(in.uv.y);
    let alpha = 1.0 - smoothstep(0.8, 1.0, d);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}