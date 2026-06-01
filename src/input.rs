pub struct InputState {
    pub is_space_pressed: bool,
    pub is_ctrl_pressed: bool,
    pub is_shift_pressed: bool,
    pub is_dragging: bool,
    pub last_mouse_pos: Option<(f64, f64)>,
    pub last_space_press_time: Option<std::time::Instant>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            is_space_pressed: false,
            is_ctrl_pressed: false,
            is_shift_pressed: false,
            is_dragging: false,
            last_mouse_pos: None,
            last_space_press_time: None,
        }
    }
}
