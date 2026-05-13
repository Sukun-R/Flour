use crate::animation::Smoother;
use winit::{event::*, keyboard::KeyCode};

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub zoom: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0,0.0,0.0,0.0),
    cgmath::Vector4::new(0.0,1.0,0.0,0.0),
    cgmath::Vector4::new(0.0,0.0,0.5,0.0),
    cgmath::Vector4::new(0.0,0.0,0.5,1.0),
);

impl Camera {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let width = self.aspect * self.zoom;
        let height = self.zoom;

        let proj = cgmath::ortho(-width, width, -height, height, self.znear, self.zfar);

        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

pub struct CameraController {
    pub speed: f32,
    pub is_forward_pressed: bool,
    pub is_backward_pressed: bool,
    pub is_left_pressed: bool,
    pub is_right_pressed: bool,
    pub zoom_target: f32,
    pub zoom_smoother: Smoother,
}

impl CameraController {
    pub fn new(speed: f32, camera: &Camera) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            zoom_target: camera.zoom,
            zoom_smoother: Smoother::new(camera.zoom, 0.25),
        }
    }

    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn handle_wheel(&mut self, upward_scroll: bool) {
        if upward_scroll {
            self.zoom_target /= 1.1;
        } else {
            self.zoom_target *= 1.1;
        }
        self.zoom_smoother.set_friction(0.25);
        self.zoom_smoother.set_target(self.zoom_target);
    }

    pub fn handle_mouse_drag(
        &mut self,
        dx: f64,
        dy: f64,
        window_size: (u32, u32),
        zoom: f32,
        camera: &mut Camera,
    ) {
        let aspect = window_size.0 as f32 / window_size.1 as f32;

        let world_dx = (dx as f32 / window_size.0 as f32) * 2.0 * aspect * zoom;
        let world_dy = (dy as f32 / window_size.1 as f32) * 2.0 * zoom;

        camera.eye.x -= world_dx;
        camera.eye.y += world_dy;
        camera.target.x -= world_dx;
        camera.target.y += world_dy;
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        use cgmath::InnerSpace;

        let side_vector = (camera.target - camera.eye).normalize().cross(camera.up);
        let up_vector = camera.up.normalize();

        let mut move_vec = cgmath::Vector3::new(0.0, 0.0, 0.0);

        if self.is_forward_pressed {
            move_vec += up_vector;
        }
        if self.is_backward_pressed {
            move_vec -= up_vector;
        }
        if self.is_left_pressed {
            move_vec -= side_vector;
        }
        if self.is_right_pressed {
            move_vec += side_vector;
        }

        if move_vec.magnitude() > 0.0 {
            let displacement = move_vec.normalize() * self.speed;
            camera.eye += displacement;
            camera.target += displacement;
        }

        self.zoom_smoother.update();
        camera.zoom = self.zoom_smoother.current;
    }
}
