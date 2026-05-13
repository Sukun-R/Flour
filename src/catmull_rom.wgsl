struct InterpPoint {
    pos: vec2<f32>,
    stroke_id: u32,
    _pad: u32,
}

struct StrokeInfo {
    color: vec4<f32>,
    width: f32,
    _pad: array<f32, 3>,
}

struct Params {
    point_count: u32,
    subdivisions: u32,
};

@group(0) @binding(0) var<storage, read> raw_points: array<InterpPoint>;
@group(0) @binding(1) var<storage, read_write> out_points: array<InterpPoint>;
@group(0) @binding(2) var<uniform> params: Params;

fn catmull_rom(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

@compute @workgroup_size(64)
fn cs_catmull(@builtin(global_invocation_id) id: vec3<u32>) {
    let seg_idx = id.x;
    let n = params.point_count;
    let sub = params.subdivisions;

    if seg_idx >= n - 1 { return; }

    let cur_id = raw_points[seg_idx].stroke_id;
    let next_id = raw_points[seg_idx + 1u].stroke_id;

    if cur_id != next_id {
        // 境界をまたぐ補間点をゼロで埋める
        for (var s = 0u; s < sub; s++) {
            out_points[seg_idx * sub + s] = InterpPoint(vec2(0.0), 0xFFFFFFFFu, 0u);
        }
        return;
    }

    let i0 = select(seg_idx, seg_idx - 1u, seg_idx > 0u && raw_points[seg_idx - 1u].stroke_id == cur_id);
    let i1 = seg_idx;
    let i2 = seg_idx + 1u;
    let i3 = select(seg_idx + 1u, seg_idx + 2u, seg_idx + 2u < n && raw_points[seg_idx + 2u].stroke_id == cur_id);

    let p0 = raw_points[i0].pos;
    let p1 = raw_points[i1].pos;
    let p2 = raw_points[i2].pos;
    let p3 = raw_points[i3].pos;

    for (var s = 0u; s < sub; s++) {
        let t = f32(s) / f32(sub);
        let pos = catmull_rom(p0, p1, p2, p3, t);
        out_points[seg_idx * sub + s] = InterpPoint(pos, cur_id, 0u);
    }

    if seg_idx == n - 2u || (seg_idx + 2u < n && raw_points[seg_idx + 2u].stroke_id != cur_id) {
        out_points[(seg_idx + 1u) * sub] = InterpPoint(p2, cur_id, 0u);
    }
}