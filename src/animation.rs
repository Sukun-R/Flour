pub struct Smoother {
    pub current: f32,
    pub target: f32,
    pub friction: f32,
}

impl Smoother {
    pub fn new(initial: f32, friction: f32) -> Self {
        Self {
            current: initial,
            target: initial,
            friction,
        }
    }

    pub fn update(&mut self) {
        let diff = self.target - self.current;
        self.current += diff * self.friction;
    }

    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }
}
