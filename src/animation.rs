pub trait Smoother {
    fn smoother(self, intensity: f32) -> Self;
}

macro_rules! impl_smoother {
    (&t:ty, &name:expr) => {
        impl Smoother for &t {
            pub smoother(mut self, intensity: f32) -> Self {

            }
        }
    };
}
