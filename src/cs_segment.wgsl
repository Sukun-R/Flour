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

struct SegmentInstance {
    point_prev: vec2<f32>,
    point_a: vec2<f32>,
    point_b: vec2<f32>,
    point_next: vec2<f32>,
    color: vec4<f32>,
    width: f32,
    _pad: array<f32, 3>,
}

struct Params {
    point_count: u32,
    subdivisions: u32,
}

@group(0) @binding(0) var<storage, read> interp_points: array<InterpPoint>;
@group(0) @binding(1) var<storage, read_write> out_segments: array<SegmentInstance>;
@group(0) @binding(2) var<storage, read> stroke_infos: array<StrokeInfo>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn cs_segment(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = (params.point_count - 1) * params.subdivisions + 1;

    if i >= n - 1u { return; }

    let cur = interp_points[i];
    let nxt = interp_points[i + 1u];

    // ストローク境界はスキップ
    if cur.stroke_id != nxt.stroke_id {
        out_segments[i] = SegmentInstance(
            vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0),
            vec4(0.0), 0.0, array<f32,3>(0.0, 0.0, 0.0),
        );
        return;
    }

    let info = stroke_infos[cur.stroke_id];

    let prev_idx = select(i, i - 1u, i > 0u);
    let next_idx = select(i + 1u, i + 2u, i + 2u < n);

    // 前後の点（境界をまたがない）
    let use_prev = i > 0u && interp_points[prev_idx].stroke_id == cur.stroke_id;
    let prev_pos = select(cur.pos, interp_points[prev_idx].pos, use_prev);

    let use_next = i + 2u < n && interp_points[next_idx].stroke_id == cur.stroke_id;
    let next_pos = select(nxt.pos, interp_points[next_idx].pos, use_next);

    out_segments[i] = SegmentInstance(
        prev_pos,
        cur.pos,
        nxt.pos,
        next_pos,
        info.color,
        info.width,
        array<f32, 3>(0.0, 0.0, 0.0),
    );
}