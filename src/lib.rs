use crate::math::{add2, dot2, normalize2, sub2};
use crate::stroke::DiscInstance;
use crate::stroke::SegmentInstance;
use crate::stroke::Stroke;

use std::sync::Arc;

use image::GenericImageView;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use wgpu::util::DeviceExt;

use cgmath::prelude::*;

mod animation;
mod camera;
mod math;
mod stroke;
mod texture;

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}
// InterpPointの対応する構造体
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InterpPoint {
    pos: [f32; 2],
    stroke_id: u32,
    _pad: u32,
}

// StrokeInfoの対応する構造体
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StrokeInfo {
    color: [f32; 4],
    width: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeParams {
    point_count: u32,
    subdivisions: u32,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
        }
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 1000;
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
    0.0,
);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

const VERTICES: &[Vertex] = &[
    // Changed
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        tex_coords: [0.4131759, 0.00759614],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        tex_coords: [0.0048659444, 0.43041354],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        tex_coords: [0.28081453, 0.949397],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        tex_coords: [0.85967, 0.84732914],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        tex_coords: [0.9414737, 0.2652641],
    }, // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub struct State {
    debug_mode: bool,
    debug_point_buffer: wgpu::Buffer,
    debug_point_count: u32,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    camera: camera::Camera,
    camera_uniform: camera::CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: camera::CameraController,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    is_space_pressed: bool,
    is_dragging: bool,
    last_mouse_pos: Option<(f64, f64)>,
    strokes: Vec<Stroke>,
    current_stroke: Option<Stroke>,
    stroke_instance_count: u32,
    stroke_buffer: wgpu::Buffer,
    stroke_pipeline: wgpu::RenderPipeline,
    disc_instance_count: u32,
    disc_buffer: wgpu::Buffer,
    disc_pipeline: wgpu::RenderPipeline,
    raw_point_buffer: wgpu::Buffer,
    interpolated_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    stroke_info_buffer: wgpu::Buffer,
    segment_compute_pipeline: wgpu::ComputePipeline,
    segment_compute_bind_group: wgpu::BindGroup,
    raw_point_count: u32,
    last_space_press_time: Option<std::time::Instant>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let debug_point_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Point Buffer"),
            size: (std::mem::size_of::<DiscInstance>() * 100_000) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let diffuse_bytes = include_bytes!("donald.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "donald.png").unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let camera = camera::Camera {
            eye: (0.0, 0.0, 10.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            zoom: 1.0,
            znear: 0.1,
            zfar: 10000.0,
        };
        let mut camera_uniform = camera::CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });
        let camera_controller = camera::CameraController::new(0.2, &camera);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&texture_bind_group_layout),
                    Some(&camera_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|y| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let position = cgmath::Vector3 {
                        x: x as f32,
                        y: y as f32,
                        z: 0.0,
                    } - INSTANCE_DISPLACEMENT;

                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(0.0))
                    };

                    Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let stroke_shader = device.create_shader_module(wgpu::include_wgsl!("stroke.wgsl"));

        let stroke_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Stroke Pipeline Layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout)],
                immediate_size: 0,
            });

        let stroke_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Stroke Pipeline"),
            layout: Some(&stroke_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &stroke_shader,
                entry_point: Some("vs_stroke"),
                buffers: &[SegmentInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &stroke_shader,
                entry_point: Some("fs_stroke"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let stroke_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stroke Buffer"),
            size: (std::mem::size_of::<SegmentInstance>() * 100_000) as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let disc_shader = device.create_shader_module(wgpu::include_wgsl!("disc.wgsl"));

        let disc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Disc Pipeline Layout"),
            bind_group_layouts: &[Some(&camera_bind_group_layout)],
            immediate_size: 0,
        });

        let disc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Disc Pipeline"),
            layout: Some(&disc_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &disc_shader,
                entry_point: Some("vs_disc"),
                buffers: &[DiscInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &disc_shader,
                entry_point: Some("fs_disc"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let disc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Disc Buffer"),
            size: (std::mem::size_of::<DiscInstance>() * 100_000) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("catmull_rom.wgsl"));

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Catmull-Rom Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("cs_catmull"),
            compilation_options: Default::default(),
            cache: None,
        });

        let max_points = 500_000usize;
        let subdivisions = 8u32;

        let raw_point_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Raw Point Buffer"),
            size: (std::mem::size_of::<InterpPoint>() * max_points) as u64, // ← [f32;2]からInterpPointに
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let interpolated_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Interpolated Buffer"),
            size: (std::mem::size_of::<InterpPoint>() * max_points * subdivisions as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE, // COPY_SRCは不要
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Params"),
            contents: bytemuck::cast_slice(&[ComputeParams {
                point_count: 0,
                subdivisions,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: interpolated_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let stroke_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stroke Info Buffer"),
            size: (std::mem::size_of::<StrokeInfo>() * 10_000) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let segment_compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("cs_segment.wgsl"));

        let segment_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Segment Compute BGL"),
                entries: &[
                    // interp_points (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // out_segments (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // stroke_infos (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let segment_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Segment Compute Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&segment_bind_group_layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &segment_compute_shader,
                entry_point: Some("cs_segment"),
                compilation_options: Default::default(),
                cache: None,
            });

        let segment_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Segment Compute BG"),
            layout: &segment_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: interpolated_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stroke_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: stroke_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            debug_mode: false,
            debug_point_buffer,   // ← 追加
            debug_point_count: 0, // ← 追加
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            instances,
            instance_buffer,
            is_space_pressed: false,
            is_dragging: false,
            last_mouse_pos: None,
            strokes: Vec::new(),
            current_stroke: None,
            stroke_instance_count: 0,
            stroke_buffer,
            stroke_pipeline,
            disc_instance_count: 0,
            disc_buffer,
            disc_pipeline,
            raw_point_buffer,
            interpolated_buffer,
            compute_pipeline,
            compute_bind_group,
            params_buffer,
            raw_point_count: 0,
            stroke_info_buffer,
            segment_compute_pipeline,
            segment_compute_bind_group,
            last_space_press_time: None,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // ① InterpPoint形式で生点群を送る（古い[f32;2]のコードは削除）
        self.upload_raw_points();
        self.upload_debug_points(); // ← 追加
        if self.raw_point_count < 2 {
            self.stroke_instance_count = 0;
            return;
        }

        // paramsを更新
        self.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[ComputeParams {
                point_count: self.raw_point_count,
                subdivisions: 8,
            }]),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Catmull-Rom"),
                timestamp_writes: None,
            });
            // ① Catmull-Rom補間
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group, &[]);
            pass.dispatch_workgroups((self.raw_point_count - 1).div_ceil(64), 1, 1);

            // ② SegmentInstance生成
            let interp_count = self.raw_point_count * 8;
            pass.set_pipeline(&self.segment_compute_pipeline);
            pass.set_bind_group(0, &self.segment_compute_bind_group, &[]);
            pass.dispatch_workgroups((interp_count - 1).div_ceil(64), 1, 1);
            self.stroke_instance_count = interp_count - 1;
        }
        self.queue.submit([encoder.finish()]);
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.surface.configure(&self.device, &self.config);
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                anyhow::bail!("Lost device");
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as _);
            if self.debug_mode {
                if self.debug_point_count > 0 {
                    render_pass.set_pipeline(&self.disc_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.debug_point_buffer.slice(..));
                    render_pass.draw(0..6, 0..self.debug_point_count);
                }
            } else {
                if self.stroke_instance_count > 0 {
                    render_pass.set_pipeline(&self.stroke_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.stroke_buffer.slice(..));
                    render_pass.draw(0..6, 0..self.stroke_instance_count);
                }

                if self.disc_instance_count > 0 {
                    render_pass.set_pipeline(&self.disc_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.disc_buffer.slice(..));
                    render_pass.draw(0..6, 0..self.disc_instance_count);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    // impl State
    fn handle_key(&mut self, _event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => self.debug_mode = !self.debug_mode,
            (KeyCode::Space, true) => {
                if !self.is_space_pressed {
                    self.is_space_pressed = true;
                    let now = std::time::Instant::now();
                    let is_double = self
                        .last_space_press_time
                        .map(|t| t.elapsed().as_millis() < 300)
                        .unwrap_or(false);

                    if is_double {
                        self.camera.zoom = 1.0;
                        self.camera_controller.zoom_target = 1.0;
                        self.camera_controller.zoom_smoother.set_friction(0.1);
                        self.camera_controller.zoom_smoother.set_target(1.0);
                        self.last_space_press_time = None;
                    } else {
                        self.last_space_press_time = Some(now);
                    }
                }
            }
            (KeyCode::Space, false) => {
                self.is_space_pressed = false;
            }

            _ => {
                self.camera_controller.handle_key(code, is_pressed);
            }
        }
    }

    fn handle_wheel(&mut self, _event_loop: &ActiveEventLoop, delta: &MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(_, y) => {
                self.camera_controller.handle_wheel(*y > 0.0);
            }
            _ => {}
        }
    }

    fn screen_to_world(&self, sx: f64, sy: f64) -> [f32; 2] {
        let size = self.window.inner_size();
        let ndc_x = (sx as f32 / size.width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (sy as f32 / size.height as f32) * 2.0;
        let wx = ndc_x * self.camera.aspect * self.camera.zoom + self.camera.eye.x;
        let wy = ndc_y * self.camera.zoom + self.camera.eye.y;
        [wx, wy]
    }
    fn rebuild_stroke_buffer(&mut self) {
        // SegmentはComputeがやるのでDiscだけ
        let mut discs: Vec<DiscInstance> = Vec::new();

        for stroke in self.strokes.iter().chain(self.current_stroke.iter()) {
            let pts = &stroke.points;
            if pts.len() < 2 {
                continue;
            }

            discs.push(DiscInstance {
                center: pts[0],
                color: stroke.color,
                width: stroke.width,
                _pad: [0.0],
            });
            discs.push(DiscInstance {
                center: pts[pts.len() - 1],
                color: stroke.color,
                width: stroke.width,
                _pad: [0.0],
            });

            for i in 1..pts.len() - 1 {
                let dir_ab = normalize2(sub2(pts[i + 1], pts[i]));
                let dir_prev = normalize2(sub2(pts[i], pts[i - 1]));
                let nor_ab = [-dir_ab[1], dir_ab[0]];
                let nor_prev = [-dir_prev[1], dir_prev[0]];
                let miter = normalize2(add2(nor_ab, nor_prev));
                let d = dot2(miter, nor_ab);
                if d > 0.0001 && 1.0 / d > 2.0 {
                    discs.push(DiscInstance {
                        center: pts[i],
                        color: stroke.color,
                        width: stroke.width,
                        _pad: [0.0],
                    });
                }
            }
        }

        self.queue
            .write_buffer(&self.disc_buffer, 0, bytemuck::cast_slice(&discs));
        self.disc_instance_count = discs.len() as u32;
    }
    fn upload_raw_points(&mut self) {
        let mut points: Vec<InterpPoint> = Vec::new();
        let mut infos: Vec<StrokeInfo> = Vec::new();
        let max_points = 500_000usize;

        for (id, stroke) in self
            .strokes
            .iter()
            .chain(self.current_stroke.iter())
            .filter(|s| s.points.len() >= 2) // ← ここでフィルタ
            .enumerate()
        {
            if points.len() >= max_points {
                break; // これ以上追加しない
            }
            if stroke.points.len() < 2 {
                continue;
            }
            infos.push(StrokeInfo {
                color: stroke.color,
                width: stroke.width,
                _pad: [0.0; 3],
            });

            let resampled = Self::resample_stroke(&stroke.points, stroke.width * 2.0);

            for &p in &resampled {
                points.push(InterpPoint {
                    pos: p,
                    stroke_id: id as u32,
                    _pad: 0,
                });
            }
        }

        self.queue
            .write_buffer(&self.raw_point_buffer, 0, bytemuck::cast_slice(&points));
        self.queue
            .write_buffer(&self.stroke_info_buffer, 0, bytemuck::cast_slice(&infos));
        self.raw_point_count = points.len() as u32;
    }

    fn resample_stroke(points: &[[f32; 2]], interval: f32) -> Vec<[f32; 2]> {
        if points.len() < 2 {
            return points.to_vec();
        }

        let mut result = vec![points[0]];
        let mut accumulated = 0.0f32;

        for i in 1..points.len() {
            let prev = points[i - 1];
            let cur = points[i];
            let dx = cur[0] - prev[0];
            let dy = cur[1] - prev[1];
            let seg_len = (dx * dx + dy * dy).sqrt();

            let mut t = (interval - accumulated) / seg_len;

            while t <= 1.0 {
                let x = prev[0] + dx * t;
                let y = prev[1] + dy * t;
                result.push([x, y]);
                t += interval / seg_len;
            }

            accumulated = (1.0 - (t - interval / seg_len)) * seg_len;
        }

        result.push(*points.last().unwrap());
        result
    }

    fn upload_debug_points(&mut self) {
        let discs: Vec<DiscInstance> = self
            .strokes
            .iter()
            .chain(self.current_stroke.iter())
            .filter(|s| s.points.len() >= 2)
            .flat_map(|stroke| {
                let resampled = Self::resample_stroke(&stroke.points, stroke.width * 2.0);
                resampled.into_iter().map(move |p| DiscInstance {
                    center: p,
                    color: [1.0, 0.0, 0.0, 0.5], // 赤で表示
                    width: stroke.width * 0.3,
                    _pad: [0.0],
                })
            })
            .collect();

        self.queue
            .write_buffer(&self.debug_point_buffer, 0, bytemuck::cast_slice(&discs));
        self.debug_point_count = discs.len() as u32;
    }
}

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use winit::platform::windows::WindowAttributesExtWindows;

        let icon_bytes = include_bytes!("gurakoro.png");
        let icon_image = image::load_from_memory(icon_bytes).unwrap();
        let icon_rgba = icon_image.to_rgba8();
        let (width, height) = icon_image.dimensions();

        let icon = winit::window::Icon::from_rgba(icon_rgba.into_raw(), width, height).unwrap();

        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes()
            .with_window_icon(Some(icon.clone()))
            .with_taskbar_icon(Some(icon))
            .with_visible(false);

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(State::new(window.clone())).unwrap());
        window.set_visible(true);
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("{e}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::MouseWheel { delta, .. } => state.handle_wheel(event_loop, &delta),
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                #[allow(clippy::option_map_unit_fn)]
                self.state.as_mut().map(|s| {
                    s.is_dragging = state.is_pressed();
                    if s.is_dragging {
                        s.current_stroke = Some(Stroke::new([0.0, 0.0, 0.0, 1.0], 0.00390625));
                        // 0.00390625
                    } else {
                        s.last_mouse_pos = None;

                        if let Some(st) = s.current_stroke.take() {
                            s.strokes.push(st);
                            s.rebuild_stroke_buffer();
                        }
                    }
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(s) = &mut self.state {
                    let cur = (position.x, position.y);
                    if s.is_dragging && s.is_space_pressed {
                        if let Some((lx, ly)) = s.last_mouse_pos {
                            let dx = cur.0 - lx;
                            let dy = cur.1 - ly;
                            let win = s.window.inner_size();
                            let zoom = s.camera.zoom;
                            s.camera_controller.handle_mouse_drag(
                                dx,
                                dy,
                                (win.width, win.height),
                                zoom,
                                &mut s.camera,
                            );
                        }
                    } else if s.is_dragging {
                        let world = s.screen_to_world(cur.0, cur.1);
                        if let Some(stroke) = &mut s.current_stroke {
                            stroke.add_point(world[0], world[1]);
                            s.rebuild_stroke_buffer();
                        }
                    }
                    s.last_mouse_pos = Some(cur);
                }
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;

    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
