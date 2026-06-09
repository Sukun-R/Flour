struct BezierSegment {
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    stroke_id: u32,
    _pad: array<u32, 3>,
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

struct StrokeInfo {
    color: vec4<f32>,
    width: f32,
    _pad: array<f32, 3>,
}

struct Params {
    segment_count: u32,
    subdivisions: u32,
}

@group(0) @binding(0) var<storage, read> bezier_segments: array<BezierSegment>;
@group(0) @binding(1) var<storage, read_write> out_segments: array<SegmentInstance>;
@group(0) @binding(2) var<storage, read> stroke_infos: array<StrokeInfo>;
@group(0) @binding(3) var<uniform> params: Params;

fn eval_bezier(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let s = 1.0 - t;
    return s * s * s * p0 + 3.0 * s * s * t * p1 + 3.0 * s * t * t * p2 + t * t * t * p3;
}

@compute @workgroup_size(64)
fn cs_bezier(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let sub = params.subdivisions;
    let total = params.segment_count * sub;

    if i >= total { return; }

    let seg_idx = i / sub;
    let sub_idx = i % sub;

    let safe_sub_idx_prev = select(sub_idx - 1u, 0u, sub_idx == 0u);
    let safe_seg_idx_prev = select(seg_idx - 1u, 0u, seg_idx == 0u);
    let safe_seg_idx_next = select(seg_idx + 1u, seg_idx, seg_idx + 1u >= params.segment_count);

    let seg = bezier_segments[seg_idx];
    let info = stroke_infos[seg.stroke_id];

    let t_a = f32(sub_idx) / f32(sub);
    let t_b = f32(sub_idx + 1u) / f32(sub);

    let point_a = eval_bezier(seg.p0, seg.p1, seg.p2, seg.p3, t_a);
    let point_b = eval_bezier(seg.p0, seg.p1, seg.p2, seg.p3, t_b);

    // point_prevとpoint_nextをシンプルに求める
    var point_prev: vec2<f32>;
    var point_next: vec2<f32>;

    // point_prev
    if sub_idx > 0u {
        point_prev = eval_bezier(seg.p0, seg.p1, seg.p2, seg.p3, f32(sub_idx - 1u) / f32(sub));
    } else if seg_idx > 0u && bezier_segments[safe_seg_idx_prev].stroke_id == seg.stroke_id {
        let prev_seg = bezier_segments[safe_seg_idx_prev];
        point_prev = eval_bezier(prev_seg.p0, prev_seg.p1, prev_seg.p2, prev_seg.p3, f32(sub - 1u) / f32(sub));
    } else {
        point_prev = point_a;
    }

    // point_next
    if sub_idx + 1u < sub {
        point_next = eval_bezier(seg.p0, seg.p1, seg.p2, seg.p3, f32(sub_idx + 2u) / f32(sub));
    } else if seg_idx + 1u < params.segment_count && bezier_segments[safe_seg_idx_next].stroke_id == seg.stroke_id {
        let next_seg = bezier_segments[safe_seg_idx_next];
        point_next = eval_bezier(next_seg.p0, next_seg.p1, next_seg.p2, next_seg.p3, 1.0 / f32(sub));
    } else {
        point_next = point_b;
    }

    out_segments[i] = SegmentInstance(
        point_prev, point_a, point_b, point_next, info.color, info.width,
        array<f32, 3>(0.0, 0.0, 0.0),
    );
}