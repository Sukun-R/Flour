use crate::math::{add2, dot2, normalize2, sub2};
use crate::renderer::{ComputeParams, InterpPoint, StrokeInfo};
use crate::renderer::{ComputeResources, DiscRenderer, StrokeRenderer};
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

mod animation;
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
    stroke_renderer: StrokeRenderer,
    disc_renderer: DiscRenderer,
    compute_resources: ComputeResources,
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
            current_stroke: None,
            stroke_renderer,
            disc_renderer,
            compute_resources,
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

        // ① InterpPoint形式で生点群を送る（古い[f32;2]のコードは削除）
        self.upload_raw_points();
        self.upload_debug_points(); // ← 追加
        if self.compute_resources.raw_point_count < 2 {
            self.stroke_renderer.instance_count = 0;
            return;
        }

        // paramsを更新
        self.queue.write_buffer(
            &self.compute_resources.params_buffer,
            0,
            bytemuck::cast_slice(&[ComputeParams {
                point_count: self.compute_resources.raw_point_count,
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
            pass.set_pipeline(&self.compute_resources.compute_pipeline);
            pass.set_bind_group(0, &self.compute_resources.compute_bind_group, &[]);
            pass.dispatch_workgroups(
                (self.compute_resources.raw_point_count - 1).div_ceil(64),
                1,
                1,
            );

            // ② SegmentInstance生成
            let interp_count = self.compute_resources.raw_point_count * 8;
            pass.set_pipeline(&self.compute_resources.segment_compute_pipeline);
            pass.set_bind_group(0, &self.compute_resources.segment_compute_bind_group, &[]);
            pass.dispatch_workgroups((interp_count - 1).div_ceil(64), 1, 1);
            self.stroke_renderer.instance_count = interp_count - 1;
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
            .write_buffer(&self.disc_renderer.buffer, 0, bytemuck::cast_slice(&discs));
        self.disc_renderer.instance_count = discs.len() as u32;
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
                    s.input.is_dragging = state.is_pressed();
                    if s.input.is_dragging {
                        s.current_stroke = Some(Stroke::new([0.0, 0.0, 0.0, 1.0], 0.00390625));
                        // 0.00390625
                    } else {
                        s.input.last_mouse_pos = None;

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
                            s.rebuild_stroke_buffer();
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
