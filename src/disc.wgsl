struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct DiscInstance {
    @location(0) center: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) width: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_disc(@builtin(vertex_index) vid: u32, inst: DiscInstance) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, 1.0),
    );

    let c = corners[vid];
    let pos2d = inst.center + c * inst.width;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos2d, 0.0, 1.0);
    out.color = inst.color;
    out.uv = c;
    return out;
}

@fragment
fn fs_disc(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    let alpha = 1.0 - smoothstep(0.9, 1.0, d);
    if alpha <= 0.0 {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}