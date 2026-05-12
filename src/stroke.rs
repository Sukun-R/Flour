#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SegmentInstance {
    pub point_prev: [f32; 2],
    pub point_a: [f32; 2],
    pub point_b: [f32; 2],
    pub point_next: [f32; 2],
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
                // point_prev
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // point_a
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // point_b
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // point_next
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // width
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DiscInstance {
    pub center: [f32; 2],
    pub color: [f32; 4],
    pub width: f32,
    pub _pad: [f32; 1],
}

impl DiscInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<DiscInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // center
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // width
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
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

            if dist_sq < self.width * 0.001 {
                return;
            }
        }
        self.points.push([x, y]);
    }
}
