use glam::Vec2;
use rand::prelude::*;

#[derive(Clone)]
pub struct Sim {
}

impl Sim {
    pub fn new(
    ) -> Self {
        let mut rng = rand::thread_rng();

        Sim {
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
    ) {
    }
}
