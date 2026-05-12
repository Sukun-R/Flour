struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct SegmentInstance {
    @location(0) point_a: vec2<f32>,
    @location(1) point_b: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) width: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_stroke(
    @builtin(vertex_index) vid: u32,
    inst: SegmentInstance,
) -> VertexOutput {
    let dir = normalize(inst.point_b - inst.point_a);
    let nor = vec2<f32>(-dir.y, dir.x);

    let signs = array<vec2<f32>, 6>(
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, -1.0),
    );

    let s = signs[vid];
    let base = mix(inst.point_a, inst.point_b, (s.x + 1.0) * 0.5);
    let pos2d = base + nor * inst.width * s.y;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos2d, 0.0, 1.0);
    out.color = inst.color;
    out.uv = s;
    return out;
}

@fragment
fn fs_stroke(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = abs(in.uv.y);
    let alpha = 1.0 - smoothstep(0.8, 1.0, d);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}