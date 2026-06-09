use crate::bezier_fit::fit_curve;
use crate::math::{add2, dot2, normalize2, sub2};
use crate::renderer::{
    BezierComputeResources, BezierParams, BezierRenderer, BezierSegment, ComputeParams,
    ComputeResources, DiscRenderer, InterpPoint, StrokeInfo, StrokeRenderer,
};
use crate::stroke::DiscInstance;
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

mod animation;
mod bezier_fit;
mod camera;
mod input;
mod math;
mod renderer;
mod stroke;
mod texture;

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
    camera: camera::CameraState,
    input: input::InputState,
    strokes: Vec<Stroke>,
    current_stroke: Option<Stroke>,
    undo_stack: Vec<Stroke>,
    stroke_renderer: StrokeRenderer,
    disc_renderer: DiscRenderer,
    compute_resources: ComputeResources,
    bezier_renderer: BezierRenderer,
    bezier_compute: BezierComputeResources,
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
        let camera = camera::CameraState::new(&device, &config);

        let input = input::InputState::new();

        let debug_point_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Point Buffer"),
            size: (std::mem::size_of::<DiscInstance>() * 100_000) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stroke_renderer = StrokeRenderer::new(&device, &config, &camera);

        let disc_renderer = DiscRenderer::new(&device, &config, &camera);

        let compute_resources = ComputeResources::new(&device, &config, &camera, &stroke_renderer);

        let bezier_renderer = BezierRenderer::new(&device, &config, &camera);

        let bezier_compute = BezierComputeResources::new(&device, &bezier_renderer);
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
            camera,
            input,
            strokes: Vec::new(),
            undo_stack: Vec::new(),
            current_stroke: None,
            stroke_renderer,
            disc_renderer,
            compute_resources,
            bezier_renderer,
            bezier_compute,
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
        self.camera
            .controller
            .update_camera(&mut self.camera.params);
        self.camera.uniform.update_view_proj(&self.camera.params);
        self.queue.write_buffer(
            &self.camera.buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );

        self.upload_points();

        // write_buffer は encoder の前に
        let interp_count = if self.compute_resources.raw_point_count >= 2 {
            self.queue.write_buffer(
                &self.compute_resources.params_buffer,
                0,
                bytemuck::cast_slice(&[ComputeParams {
                    point_count: self.compute_resources.raw_point_count,
                    subdivisions: 8,
                }]),
            );
            (self.compute_resources.raw_point_count - 1) * 8 + 1
        } else {
            0
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            if self.bezier_compute.segment_count > 0 {
                pass.set_pipeline(&self.bezier_compute.compute_pipeline);
                pass.set_bind_group(0, &self.bezier_compute.compute_bind_group, &[]);
                pass.dispatch_workgroups(self.bezier_renderer.instance_count.div_ceil(64), 1, 1);
            }

            if interp_count > 0 {
                pass.set_pipeline(&self.compute_resources.compute_pipeline);
                pass.set_bind_group(0, &self.compute_resources.compute_bind_group, &[]);
                pass.dispatch_workgroups(
                    (self.compute_resources.raw_point_count - 1).div_ceil(64),
                    1,
                    1,
                );

                pass.set_pipeline(&self.compute_resources.segment_compute_pipeline);
                pass.set_bind_group(0, &self.compute_resources.segment_compute_bind_group, &[]);
                pass.dispatch_workgroups((interp_count - 1).div_ceil(64), 1, 1);

                self.stroke_renderer.instance_count = interp_count - 1;
            } else {
                self.stroke_renderer.instance_count = 0;
            }
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

            if self.debug_mode {
                if self.debug_point_count > 0 {
                    render_pass.set_pipeline(&self.disc_renderer.pipeline);
                    render_pass.set_bind_group(0, &self.camera.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.debug_point_buffer.slice(..));
                    render_pass.draw(0..6, 0..self.debug_point_count);
                }
            } else {
                if self.stroke_renderer.instance_count > 0 {
                    render_pass.set_pipeline(&self.stroke_renderer.pipeline);
                    render_pass.set_bind_group(0, &self.camera.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.stroke_renderer.buffer.slice(..));
                    render_pass.draw(0..6, 0..self.stroke_renderer.instance_count);
                }

                if self.disc_renderer.instance_count > 0 {
                    render_pass.set_pipeline(&self.disc_renderer.pipeline);
                    render_pass.set_bind_group(0, &self.camera.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.disc_renderer.buffer.slice(..));
                    render_pass.draw(0..6, 0..self.disc_renderer.instance_count);
                }

                if self.bezier_renderer.instance_count > 0 {
                    render_pass.set_pipeline(&self.bezier_renderer.pipeline);
                    render_pass.set_bind_group(0, &self.camera.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.bezier_renderer.buffer.slice(..));
                    render_pass.draw(0..6, 0..self.bezier_renderer.instance_count);
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
                if !self.input.is_space_pressed {
                    self.input.is_space_pressed = true;
                    let now = std::time::Instant::now();
                    let is_double = self
                        .input
                        .last_space_press_time
                        .map(|t| t.elapsed().as_millis() < 300)
                        .unwrap_or(false);

                    if is_double {
                        self.camera.params.zoom = 1.0;
                        self.camera.controller.zoom_target = 1.0;
                        self.camera.controller.zoom_smoother.set_friction(0.1);
                        self.camera.controller.zoom_smoother.set_target(1.0);
                        self.input.last_space_press_time = None;
                    } else {
                        self.input.last_space_press_time = Some(now);
                    }
                }
            }
            (KeyCode::Space, false) => {
                self.input.is_space_pressed = false;
            }
            (KeyCode::ControlLeft, _) | (KeyCode::ControlRight, _) => {
                self.input.is_ctrl_pressed = is_pressed;
            }
            (KeyCode::ShiftLeft, _) | (KeyCode::ShiftRight, _) => {
                self.input.is_shift_pressed = is_pressed;
            }
            (KeyCode::KeyZ, true) => {
                if !self.input.is_ctrl_pressed {
                    return;
                }

                match self.input.is_shift_pressed {
                    //Ctrl + Z
                    false => {
                        if let Some(stroke) = self.strokes.pop() {
                            self.undo_stack.push(stroke);
                            self.queue.write_buffer(
                                &self.compute_resources.interpolated_buffer,
                                0,
                                &vec![
                                    0u8;
                                    self.compute_resources.interpolated_buffer.size() as usize
                                ],
                            );
                            self.rebuild_stroke_buffer();
                        }
                    }
                    //Ctrl + Shift + Z
                    _ => {
                        if let Some(stroke) = self.undo_stack.pop() {
                            self.strokes.push(stroke);
                            self.rebuild_stroke_buffer();
                        }
                    }
                }
            }

            _ => {
                self.camera.controller.handle_key(code, is_pressed);
            }
        }
    }

    fn handle_wheel(&mut self, _event_loop: &ActiveEventLoop, delta: &MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(_, y) => {
                self.camera.controller.handle_wheel(*y > 0.0);
            }
            _ => {}
        }
    }

    fn screen_to_world(&self, sx: f64, sy: f64) -> [f32; 2] {
        let size = self.window.inner_size();
        let ndc_x = (sx as f32 / size.width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (sy as f32 / size.height as f32) * 2.0;
        let wx =
            ndc_x * self.camera.params.aspect * self.camera.params.zoom + self.camera.params.eye.x;
        let wy = ndc_y * self.camera.params.zoom + self.camera.params.eye.y;
        [wx, wy]
    }
    fn rebuild_stroke_buffer(&mut self) {
        println!("rebuild_stroke_buffer called");
        self.upload_points();
        self.upload_committed_strokes();

        // SegmentはComputeがやるのでDiscだけ
        let mut discs: Vec<DiscInstance> = Vec::new();

        for stroke in self.strokes.iter().chain(self.current_stroke.iter()) {
            if stroke.committed_points.len() >= 4 {
                let first = stroke.committed_points[0];
                let last = *stroke.committed_points.last().unwrap();

                discs.push(DiscInstance {
                    center: first,
                    color: stroke.color,
                    width: stroke.width,
                    _pad: [0.0],
                });
                discs.push(DiscInstance {
                    center: last,
                    color: stroke.color,
                    width: stroke.width,
                    _pad: [0.0],
                });

                // step_by(3) で隣接セグメントの境界点を取り出す
                let pts = &stroke.committed_points;
                let mut i = 0;
                while i + 3 < pts.len() {
                    // セグメント境界点 = pts[i+3] (= 次のセグメントのp0)
                    if i + 3 < pts.len() - 1 {
                        let p = pts[i + 3];
                        let dir_in = normalize2(sub2(pts[i + 3], pts[i + 2]));
                        let dir_out = normalize2(sub2(pts[i + 4], pts[i + 3]));
                        let nor_in = [-dir_in[1], dir_in[0]];
                        let nor_out = [-dir_out[1], dir_out[0]];
                        let miter = normalize2(add2(nor_in, nor_out));
                        let d = dot2(miter, nor_in);
                        if d > 0.0001 && 1.0 / d > 2.0 {
                            discs.push(DiscInstance {
                                center: p,
                                color: stroke.color,
                                width: stroke.width,
                                _pad: [0.0],
                            });
                        }
                    }
                    i += 3;
                }
            } else {
                let pts = Self::resample_stroke(&stroke.points, stroke.width * 2.0);
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
                        })
                    }
                }
            }
        }

        self.queue
            .write_buffer(&self.disc_renderer.buffer, 0, bytemuck::cast_slice(&discs));
        self.disc_renderer.instance_count = discs.len() as u32;
    }

    fn rebuild_stroke_buffer_drawing(&mut self) {
        self.upload_points();

        let mut discs: Vec<DiscInstance> = Vec::new();

        // 確定済みストロークのDiscも含める
        for stroke in self.strokes.iter() {
            if stroke.committed_points.len() >= 4 {
                let first = stroke.committed_points[0];
                let last = *stroke.committed_points.last().unwrap();
                discs.push(DiscInstance {
                    center: first,
                    color: stroke.color,
                    width: stroke.width,
                    _pad: [0.0],
                });
                discs.push(DiscInstance {
                    center: last,
                    color: stroke.color,
                    width: stroke.width,
                    _pad: [0.0],
                });

                let pts = &stroke.committed_points;
                let mut i = 0;
                while i + 3 < pts.len() {
                    if i + 3 < pts.len() - 1 {
                        let p = pts[i + 3];
                        let dir_in = normalize2(sub2(pts[i + 3], pts[i + 2]));
                        let dir_out = normalize2(sub2(pts[i + 4], pts[i + 3]));
                        let nor_in = [-dir_in[1], dir_in[0]];
                        let nor_out = [-dir_out[1], dir_out[0]];
                        let miter = normalize2(add2(nor_in, nor_out));
                        let d = dot2(miter, nor_in);
                        if d > 0.0001 && 1.0 / d > 2.0 {
                            discs.push(DiscInstance {
                                center: p,
                                color: stroke.color,
                                width: stroke.width,
                                _pad: [0.0],
                            });
                        }
                    }
                    i += 3;
                }
            }
        }

        // 描画中ストロークのDisc
        if let Some(stroke) = &self.current_stroke {
            let pts = Self::resample_stroke(&stroke.points, stroke.width * 2.0);
            if pts.len() >= 2 {
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
        }

        self.queue
            .write_buffer(&self.disc_renderer.buffer, 0, bytemuck::cast_slice(&discs));
        self.disc_renderer.instance_count = discs.len() as u32;
    }

    fn upload_points(&mut self) {
        let mut points: Vec<InterpPoint> = Vec::new();
        let mut infos: Vec<StrokeInfo> = Vec::new();
        let mut debug_discs: Vec<DiscInstance> = Vec::new();
        let max_points = 500_000usize;

        for (id, stroke) in self
            .current_stroke
            .iter()
            .filter(|s| s.points.len() >= 2)
            .enumerate()
        {
            if points.len() >= max_points {
                break;
            }

            let interval = stroke.width * 2.0;
            let pts = Self::resample_stroke(&stroke.points, interval);

            infos.push(StrokeInfo {
                color: stroke.color,
                width: stroke.width,
                _pad: [0.0; 3],
            });

            for &p in &pts {
                points.push(InterpPoint {
                    pos: p,
                    stroke_id: id as u32,
                    _pad: 0,
                });
                // ★ 同じ p を debug 用にも積む（resample は1回）
                debug_discs.push(DiscInstance {
                    center: p,
                    color: [1.0, 0.0, 0.0, 0.5],
                    width: stroke.width * 0.3,
                    _pad: [0.0],
                });
            }
        }

        self.queue.write_buffer(
            &self.compute_resources.raw_point_buffer,
            0,
            bytemuck::cast_slice(&points),
        );
        self.queue.write_buffer(
            &self.compute_resources.stroke_info_buffer,
            0,
            bytemuck::cast_slice(&infos),
        );
        self.compute_resources.raw_point_count = points.len() as u32;

        // debug_mode のときだけ GPU に送る
        if self.debug_mode {
            self.queue.write_buffer(
                &self.debug_point_buffer,
                0,
                bytemuck::cast_slice(&debug_discs),
            );
            self.debug_point_count = debug_discs.len() as u32;
        }
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
            if seg_len < 0.000001 {
                continue;
            }

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

    fn upload_committed_strokes(&mut self) {
        let mut segments: Vec<BezierSegment> = Vec::new();
        let mut infos: Vec<StrokeInfo> = Vec::new();
        let mut info_id = 0u32;

        for stroke in self.strokes.iter() {
            if stroke.committed_points.len() < 4 {
                continue;
            }
            println!("committed len: {}", stroke.committed_points.len());
            println!(
                "expected segments: {}",
                (stroke.committed_points.len() - 1) / 3
            );
            infos.push(StrokeInfo {
                color: stroke.color,
                width: stroke.width,
                _pad: [0.0; 3],
            });

            for i in (0..stroke.committed_points.len().saturating_sub(1)).step_by(3) {
                if i + 3 >= stroke.committed_points.len() {
                    break;
                }
                segments.push(BezierSegment {
                    p0: stroke.committed_points[i],
                    p1: stroke.committed_points[i + 1],
                    p2: stroke.committed_points[i + 2],
                    p3: stroke.committed_points[i + 3],
                    stroke_id: info_id,
                    _pad: [0; 3],
                });
            }
            info_id += 1;
        }

        if segments.is_empty() {
            self.bezier_renderer.instance_count = 0;
            return;
        }

        self.queue.write_buffer(
            &self.bezier_compute.bezier_segment_buffer,
            0,
            bytemuck::cast_slice(&segments),
        );
        self.queue.write_buffer(
            &self.bezier_compute.stroke_info_buffer,
            0,
            bytemuck::cast_slice(&infos),
        );

        let segment_count = segments.len() as u32;
        let subdivisions = 8u32;

        self.queue.write_buffer(
            &self.bezier_compute.params_buffer,
            0,
            bytemuck::cast_slice(&[BezierParams {
                segment_count,
                subdivisions,
            }]),
        );

        self.bezier_compute.segment_count = segment_count;
        self.bezier_renderer.instance_count = segment_count * subdivisions;
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
                    s.input.is_dragging = state.is_pressed();
                    if s.input.is_dragging {
                        s.current_stroke = Some(Stroke::new([0.0, 0.0, 0.0, 1.0], 0.00390625));
                        // 0.00390625
                    } else {
                        s.input.last_mouse_pos = None;

                        if let Some(mut st) = s.current_stroke.take() {
                            let error = (st.width * 0.001) * (st.width * 0.001);
                            st.committed_points = fit_curve(&st.points, error);
                            println!(
                                "segments: {}",
                                (st.committed_points.len().saturating_sub(1)) / 3
                            );
                            for chunk in (0..st.committed_points.len().saturating_sub(1)).step_by(3)
                            {
                                if chunk + 3 >= st.committed_points.len() {
                                    break;
                                }
                                println!(
                                    "seg {}: {:?} -> {:?}",
                                    chunk / 3,
                                    st.committed_points[chunk],
                                    st.committed_points[chunk + 3]
                                );
                            }
                            s.strokes.push(st);
                            s.undo_stack.clear();
                            s.rebuild_stroke_buffer();
                        }
                    }
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(s) = &mut self.state {
                    let cur = (position.x, position.y);
                    if s.input.is_dragging && s.input.is_space_pressed {
                        if let Some((lx, ly)) = s.input.last_mouse_pos {
                            let dx = cur.0 - lx;
                            let dy = cur.1 - ly;
                            let win = s.window.inner_size();
                            let zoom = s.camera.params.zoom;
                            s.camera.controller.handle_mouse_drag(
                                dx,
                                dy,
                                (win.width, win.height),
                                zoom,
                                &mut s.camera.params,
                            );
                        }
                    } else if s.input.is_dragging {
                        let world = s.screen_to_world(cur.0, cur.1);
                        if let Some(stroke) = &mut s.current_stroke {
                            stroke.add_point(world[0], world[1]);
                            s.rebuild_stroke_buffer_drawing();
                        }
                    }
                    s.input.last_mouse_pos = Some(cur);
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
