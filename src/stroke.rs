#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SegmentInstance {
    pub point_a: [f32; 2],
    pub point_b: [f32; 2],
    pub color: [f32; 4],
    pub width: f32,
    pub _pad: [f32; 3],
}

impl SegmentInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<SegmentInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

pub struct Stroke {
    pub points: Vec<[f32; 2]>, // 制御点列
    pub color: [f32; 4],
    pub width: f32,
}

impl Stroke {
    pub fn new(color: [f32; 4], width: f32) -> Self {
        Self {
            points: Vec::new(),
            color,
            width,
        }
    }

    pub fn add_point(&mut self, x: f32, y: f32) {
        if let Some(&last) = self.points.last() {
            let dx = x - last[0];
            let dy = y - last[1];

            let dist_sq = dx * dx + dy * dy;

            if dist_sq < 0.0001 {
                return;
            }
        }
        self.points.push([x, y]);
    }
}
